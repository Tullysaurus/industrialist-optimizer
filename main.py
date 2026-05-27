import json
import math
import pulp

# --- Parsing Helpers ---
def parse_mamyflux(mf_str):
    if not mf_str: return 0.0
    mf_str = str(mf_str).lower().replace('mf', '')
    multiplier = 1
    if 'k' in mf_str: multiplier = 1000
    if 'm' in mf_str: multiplier = 1000000
    if 'g' in mf_str: multiplier = 1000000000
    try:
        clean_str = ''.join(c for c in mf_str if c.isdigit() or c == '.')
        return float(clean_str) * multiplier
    except: return 0.0

def parse_items(item_str):
    if not item_str or item_str == "": return {}
    res = {}
    for part in item_str.split(';'):
        if ' x ' in part:
            name_qty = part.split(' x ')
            if len(name_qty) == 2:
                name, qty = name_qty
                name = name.strip()
                qty = qty.strip()
                if name and qty:  # Only add if both name and qty are non-empty
                    try:
                        res[name] = float(qty)
                    except ValueError:
                        continue  # Skip items with invalid quantities
    return res

def parse_banned_machines(machine_str):
    """Parse comma-separated list of machine names to ban."""
    if not machine_str or machine_str == "": return []
    machines = []
    for part in machine_str.split(';'):
        machine = part.strip()
        if machine:
            machines.append(machine)
    return machines

class IndustrialistOptimizer:
    def __init__(self, m_path, r_path):
        with open(m_path, 'r', encoding='utf-8') as f: 
            self.m_data = {m['Title']: m for m in json.load(f)}
        with open(r_path, 'r', encoding='utf-8') as f: 
            self.r_data = json.load(f)
        
        self.all_game_items = set()
        for r in self.r_data:
            r['parsed_in'] = parse_items(r['inputs'])
            r['parsed_out'] = parse_items(r['outputs'])
            r['parsed_pwr'] = parse_mamyflux(r['mamyflux'])
            raw_time = str(r.get('time', 1)).lower()
            r['clean_time'] = 1.0 if "variable" in raw_time else float(''.join(c for c in raw_time if c.isdigit() or c == '.'))
            
            m_info = self.m_data.get(r['machine'], {})
            r['cost'] = m_info.get('Cost', 0)
            r['tier'] = m_info.get('Tier', 0)
            
            # Split Size into W and H for GA 2D placement
            size_parts = m_info.get('Size', '1x1').split('x')
            r['width'] = int(size_parts[0])
            r['height'] = int(size_parts[1])
            r['area'] = r['width'] * r['height']

            r['parsed_in'] = {k: v/r['clean_time'] for k, v in r['parsed_in'].items()}
            r['parsed_out'] = {k: v/r['clean_time'] for k, v in r['parsed_out'].items()}

            self.all_game_items.update(r['parsed_in'].keys())
            self.all_game_items.update(r['parsed_out'].keys())

    def run(self, targets, optimize_for="space", banned_machines=None, min_tier=0, show_best=True):
        res = self._solve(targets, optimize_for, banned_machines, min_tier, relaxed=show_best)
        is_best = False
        if res and res.get('missing'):
            is_best = True
        return res, is_best

    def _solve(self, targets, optimize_for, banned_machines, min_tier, relaxed=False):
        prob = pulp.LpProblem("Factory_Opt", pulp.LpMinimize)
        machine_vars = [pulp.LpVariable(f"M{i}", lowBound=0) for i in range(len(self.r_data))]
        search_items = self.all_game_items.union(set(targets.keys()))
        slack_vars = {item: pulp.LpVariable(f"Slack_{hash(item)}", lowBound=0) for item in search_items}

        objective_terms = []
        for i, r in enumerate(self.r_data):
            if optimize_for == "space":
                num_pipes = len(r['parsed_in']) + len(r['parsed_out'])
                v_buffer = 1 if num_pipes <= 4 else 2
                
                # The +5 is the "Manifold Penalty"
                base_cost = (r['width'] * (r['height'] + v_buffer * 2)) + 5
            elif optimize_for == "power": base_cost = r['parsed_pwr']
            else: base_cost = 1.0
            penalty = 5000.0 if ((banned_machines and r['machine'] in banned_machines) or (r['tier'] < min_tier)) else 1.0
            objective_terms.append(machine_vars[i] * (base_cost * penalty))

        if relaxed:
            objective_terms.extend([slack_vars[item] * 1000000 for item in search_items])

        prob += pulp.lpSum(objective_terms)

        for item in search_items:
            net = []
            for i, r in enumerate(self.r_data):
                rate = (r['parsed_out'].get(item, 0) - r['parsed_in'].get(item, 0))
                if item == "Water":
                    hw_surplus = r['parsed_out'].get("Hot Water", 0)
                    if hw_surplus > 0: rate += hw_surplus
                if rate != 0: net.append(machine_vars[i] * rate)
            
            target_val = targets.get(item, 0)
            if relaxed:
                prob += pulp.lpSum(net) + slack_vars[item] >= target_val
            else:
                if net: prob += pulp.lpSum(net) >= target_val
                elif target_val > 0: return None

        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=20))
        if pulp.LpStatus[prob.status] != 'Optimal': return None

        def safe_val(v):
            val = pulp.value(v)
            return val if val is not None else 0.0

        return {
            'machines': {i: safe_val(machine_vars[i]) for i in range(len(self.r_data)) if safe_val(machine_vars[i]) > 1e-6},
            'missing': {item: safe_val(slack_vars[item]) for item in search_items if safe_val(slack_vars[item]) > 1e-6}
        }

    def save_to_ga_json(self, results, targets, filename="Out.json"):
        """Exports every individual machine instance for GA 2D packing and distance optimization."""
        if not results: return
        
        machine_instances = []
        instance_counter = 0
        
        # 1. Create unique instances for every machine
        # GA needs to place 10 separate pumps, not "1 pump x 10"
        recipe_to_instances = {} # Map recipe_index -> list of unique instance IDs
        
        for i, val in results['machines'].items():
            r = self.r_data[i]
            count = math.ceil(val - 1e-9)
            recipe_to_instances[i] = []
            
            # Calculate the portion of total output this single instance handles
            # Usually 1.0, but if the solver needs 0.5 of a machine, this is 0.5
            load_per_machine = val / count 

            for _ in range(count):
                inst_id = f"inst_{instance_counter}"
                instance_counter += 1
                recipe_to_instances[i].append(inst_id)
                
                machine_instances.append({
                    "id": inst_id,
                    "recipe_index": i,
                    "title": r['machine'],
                    "width": r['width'],
                    "height": r['height'],
                    "area": r['area'],
                    "pollution": parse_mamyflux(self.m_data.get(r['machine'], {}).get('Pollution', '0')),
                    "load_factor": load_per_machine,
                    "clean_time": r['clean_time'],
                    "inputs": r['parsed_in'],
                    "outputs": r['parsed_out'],
                    "connections": [] # We will fill this next
                })

        # 2. Map Connections (Weights for GA cost function)
        # GA should try to keep machines with high 'rate' closer together
        for inst in machine_instances:
            r_idx = inst['recipe_index']
            r = self.r_data[r_idx]
            
            for item, qty in r['parsed_out'].items():
                p_rate_total = (qty) * results['machines'][r_idx]
                
                # Find consumers
                for consumer_recipe_idx, consumer_val in results['machines'].items():
                    r_cons = self.r_data[consumer_recipe_idx]
                    if item in r_cons['parsed_in']:
                        # Calculate amount moved between these two recipe groups
                        c_rate_needed = (r_cons['parsed_in'][item]) * consumer_val
                        flow_rate = min(p_rate_total, c_rate_needed)
                        
                        # Distribute flow weight across all instances for GA logic
                        weight_per_pair = flow_rate / (len(recipe_to_instances[r_idx]) * len(recipe_to_instances[consumer_recipe_idx]))
                        
                        for target_id in recipe_to_instances[consumer_recipe_idx]:
                            inst["connections"].append({
                                "target_id": target_id,
                                "item": item,
                                "rate": round(weight_per_pair, 4)
                            })
                
                # Check if this item is a final target
                if targets and item in targets:
                    inst["connections"].append({
                        "target_id": "FINAL_OUTPUT",
                        "item": item,
                        "rate": round(p_rate_total / len(recipe_to_instances[r_idx]), 4)
                    })

        output = {
            "summary": {
                "total_instances": len(machine_instances),
                "total_area_required": sum(m['area'] for m in machine_instances),
                "missing_resources": results['missing']
            },
            "machines": machine_instances
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
        print(f"\n[+] GA Data exported to {filename}. Total machine instances: {len(machine_instances)}")

    def print_report(self, results, pipe_price=500, targets=None):
        if not results or not results.get('machines'):
            print("\n[!] Error: No solution found.")
            return
        total_cost, total_pwr, total_pol, total_m, total_area = 0, 0, 0, 0, 0
        print(f"\n{'MACHINE':<31} | {'COUNT':<10} | {'NET OUTPUT/s'}")
        print("-" * 85)
        sorted_indices = sorted(results['machines'].keys(), key=lambda k: self.r_data[k]['machine'])
        for i in sorted_indices:
            val = results['machines'][i]; r = self.r_data[i]
            m = self.m_data.get(r['machine'], {"Cost": 0, "Pollution": "0", "Area": 1})
            count = math.ceil(val - 1e-9)
            total_m += count
            total_cost += m.get('Cost', 0) * count
            total_pwr += r['parsed_pwr'] * val 
            total_area += r['area'] * count
            pol_raw = str(m.get('Pollution', '0')).lower()
            m_pol = 0.0
            if "variable" not in pol_raw:
                clean = ''.join(c for c in pol_raw if c.isdigit() or c in '.-')
                try: m_pol = (float(clean.split('-')[0]) + float(clean.split('-')[1]))/2 if '-' in clean else float(clean)
                except: m_pol = 0.0
            total_pol += m_pol * count
            out_str = ", ".join([f"{(qty)*val:.2f} {name}" for name, qty in r['parsed_out'].items()])
            print(f"{r['machine']:<31} | {count:<10} | {out_str}")
        if results['missing']:
            print("-" * 85); print("--- EXTERNAL INPUTS REQUIRED ---")
            for item, qty in results['missing'].items(): print(f" [!] SOURCE: {qty:.2f}/s of {item}")
        p_cost = total_m * 2 * pipe_price 
        print("-" * 85)
        print(f"TOTAL MACHINES: {total_m} | POWER: {total_pwr:,.2f} MF/s | POLLUTION: {total_pol:.2f}%/h")
        print(f"TOTAL SPACE: {total_area} Blocks^2 | ESTIMATED COST: ${total_cost:,}")

    def print_connection_map(self, results, targets):
        if not results or not results['machines']: return
        print("\n--- RESOURCE FLOW MAP ---")
        machine_produced = set()
        for i in results['machines']: machine_produced.update(self.r_data[i]['parsed_out'].keys())
        for i, val in results['machines'].items():
            r = self.r_data[i]
            if r['parsed_in']:
                in_list = [f"{(qty)*val:.2f}/s {item}{'' if (item in machine_produced or item=='Water') else ' (EXTERNAL)'}" for item, qty in r['parsed_in'].items()]
                print(f"\n[ {r['machine']} ] consumes {', '.join(in_list)}")
            for item, qty in r['parsed_out'].items():
                p_rate = (qty) * val
                print(f"  └──> Yields {p_rate:.2f}/s {item}")

def display_factory_layout(json_file="Out.json"):
    """Display the factory layout from Out.json in a human-readable format."""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[!] {json_file} not found. Run the optimizer first.")
        return
    
    machines = data.get('machines', [])
    if not machines:
        print("[!] No machines in factory layout.")
        return
    
    # Create a mapping of instance IDs to their machine titles
    instance_to_machine = {}
    for machine in machines:
        instance_to_machine[machine['id']] = machine['title']
    
    # Group machines by (title, inputs_signature, outputs_signature, load_factor)
    # This allows us to group identical machines together
    machine_groups = {}
    
    for machine in machines:
        title = machine['title']
        load_factor = machine['load_factor']
        
        # Create signatures for inputs and outputs (order-independent)
        inputs_sig = tuple(sorted(machine['inputs'].items()))
        outputs_sig = tuple(sorted(machine['outputs'].items()))
        
        key = (title, inputs_sig, outputs_sig, load_factor)
        
        if key not in machine_groups:
            machine_groups[key] = []
        machine_groups[key].append(machine)
    
    print("\n" + "="*100)
    print("FACTORY LAYOUT - DETAILED BREAKDOWN")
    print("="*100)
    
    # Display each group
    group_num = 1
    for (title, inputs_sig, outputs_sig, load_factor), group in machine_groups.items():
        count = len(group)
        sample_machine = group[0]
        
        # Display machine header
        width = sample_machine.get('width', 1)
        height = sample_machine.get('height', 1)
        
        if count > 1:
            print(f"\n[{group_num}] {count}x {title} ({width}x{height})")
        else:
            print(f"\n[{group_num}] {title} ({width}x{height})")
        
        if load_factor < 1.0:
            print(f"    Load Factor: {load_factor*100:.1f}%")
        
        # Display inputs
        if sample_machine['inputs']:
            print(f"    INPUTS:")
            for item_name, qty in sorted(sample_machine['inputs'].items()):
                per_machine_rate = (qty) * load_factor
                total_rate = (qty) * load_factor * count
                
                print(f"      ← {item_name}: {total_rate:.2f}/s ({per_machine_rate:.2f}/s each)")
        
        # Display outputs
        if sample_machine['outputs']:
            print(f"    OUTPUTS:")
            for item_name, qty in sorted(sample_machine['outputs'].items()):
                per_machine_rate = (qty) * load_factor
                total_rate = (qty) * load_factor * count
                
                print(f"      → {item_name}: {total_rate:.2f}/s ({per_machine_rate:.2f}/s each)")
        
        # Display connections (where outputs go)
        if sample_machine['connections']:
            print(f"    CONNECTIONS:")
            # Aggregate connections across all machines in group
            connection_data = {}  # (target_machine, item) -> list of (instance_id, rate)
            for machine_inst in group:
                for conn in machine_inst['connections']:
                    target = conn['target_id']
                    item = conn['item']
                    rate = conn['rate']
                    
                    key = (target, item)
                    if key not in connection_data:
                        connection_data[key] = []
                    connection_data[key].append((target, rate))
            
            # Calculate total output per item
            total_output_per_item = {}
            for (target, item), connections in connection_data.items():
                if item not in total_output_per_item:
                    total_output_per_item[item] = 0
                total_output_per_item[item] += sum(rate for _, rate in connections)
            
            # Group connections by target machine and item
            grouped_connections = {}  # (target_machine, item) -> (total_rate, instances, count)
            for (target, item), connections in connection_data.items():
                target_machine = instance_to_machine.get(target, target)
                total_rate = sum(rate for _, rate in connections)
                instance_ids = [target]
                
                key = (target_machine, item)
                if key not in grouped_connections:
                    grouped_connections[key] = {'rate': 0, 'instances': [], 'count': 0}
                
                grouped_connections[key]['rate'] += total_rate
                grouped_connections[key]['instances'].append(target)
                grouped_connections[key]['count'] += 1
            
            # Sort by rate descending, then by target machine name
            sorted_connections = sorted(
                grouped_connections.items(),
                key=lambda x: (-x[1]['rate'], x[0][0])
            )
            
            # Display connections
            for (target_machine, item), conn_info in sorted_connections:
                total_rate = conn_info['rate']
                instances = sorted(conn_info['instances'], key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0)
                
                # Calculate percentage
                item_total = total_output_per_item.get(item, total_rate)
                percentage = (total_rate / item_total * 100) if item_total > 0 else 0
                
                if target_machine == "FINAL_OUTPUT":
                    print(f"      ↘ {item} → FINAL OUTPUT: {total_rate:.2f}/s ({percentage:.1f}%)")
                else:
                    # Group consecutive instances
                    if len(instances) > 1:
                        # Try to group consecutive instance numbers
                        instance_nums = []
                        for inst in instances:
                            if '_' in inst and inst.split('_')[1].isdigit():
                                instance_nums.append(int(inst.split('_')[1]))
                            else:
                                instance_nums.append(-1)
                        
                        instance_nums.sort()
                        
                        # Find consecutive groups
                        groups = []
                        if instance_nums:
                            current_group = [instance_nums[0]]
                            for num in instance_nums[1:]:
                                if num == current_group[-1] + 1:
                                    current_group.append(num)
                                else:
                                    groups.append(current_group)
                                    current_group = [num]
                            groups.append(current_group)
                        
                        # Format the instance range
                        range_str = ", ".join([
                            f"inst_{g[0]}-{g[-1]}" if len(g) > 1 else f"inst_{g[0]}" 
                            for g in groups
                        ])
                        
                        print(f"      ↘ {item} → {conn_info['count']}x {target_machine} ({range_str}): {total_rate:.2f}/s ({percentage:.1f}%)")
                    else:
                        print(f"      ↘ {item} → {target_machine} ({instances[0]}): {total_rate:.2f}/s ({percentage:.1f}%)")
        
        group_num += 1
    
    # Display summary
    summary = data.get('summary', {})
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"Total Machine Instances: {summary.get('total_instances', 0)}")
    print(f"Total Area Required: {summary.get('total_area_required', 0)} blocks²")
    
    missing = summary.get('missing_resources', {})
    if missing:
        print(f"\nMissing Resources:")
        for item, qty in missing.items():
            print(f"  [!] {item}: {qty:.2f}/s (EXTERNAL INPUT REQUIRED)")
    
    print("="*100 + "\n")

if __name__ == "__main__":
    # Get user demand input
    print("Enter demand in format: Item x Count (e.g., 'Coal x 352')")
    print("Enter multiple items separated by semicolons (e.g., 'Coal x 352; Iron Ore x 100')")
    
    # Load previous valid input from file
    previous_input_file = "previous_input.txt"
    previous_valid_input = None
    try:
        with open(previous_input_file, 'r') as f:
            previous_valid_input = f.read().strip()
    except FileNotFoundError:
        pass
    
    user_input = input("\nDEMAND: ").strip()
    
    # If user input is empty, use previous valid input
    if not user_input and previous_valid_input:
        user_input = previous_valid_input
        print(f"[*] Using previous input: {user_input}")
    
    DEMAND = parse_items(user_input)

    if not DEMAND:
        print("[!] Invalid input format. Using default: Purple Gold Ingot x 0.1")
        DEMAND = {"Purple Gold Ingot": 0.1}
    else:
        # Save this as the new valid input
        with open(previous_input_file, 'w') as f:
            f.write(user_input)
    
    # Get banned machines input
    print("\nEnter machines to ban (colon-separated, e.g., 'Mineshaft Drill; Water Pump')")
    
    # Load previous banned machines from file
    previous_ban_file = "previous_bans.txt"
    previous_ban_input = None
    try:
        with open(previous_ban_file, 'r') as f:
            previous_ban_input = f.read().strip()
    except FileNotFoundError:
        pass
    
    ban_input = input("BANNED MACHINES: ").strip()
    
    # If user input is empty, use previous ban input
    if not ban_input and previous_ban_input:
        ban_input = previous_ban_input
        print(f"[*] Using previous bans: {ban_input}")
    
    banned_machines = parse_banned_machines(ban_input)
    
    if ban_input and not banned_machines:
        print("[!] Invalid ban input format. Banning nothing.")
    elif banned_machines:
        # Save this as the new valid ban input
        with open(previous_ban_file, 'w') as f:
            f.write(ban_input)
        print(f"[*] Banning: {', '.join(banned_machines)}")
    
    print(f"\n[*] Running optimization for: {DEMAND}\n")
    
    opt = IndustrialistOptimizer('constants/Machines.json', 'constants/Recipes.json')
    res, is_best = opt.run(DEMAND, optimize_for="machines", banned_machines=banned_machines, min_tier=1, show_best=False)
    
    if res:
        if is_best: print("\n*** NOTE: SHOWING BEST EFFORT BUILD (RESOURCES MISSING) ***")
        opt.print_report(res, targets=DEMAND)
        opt.save_to_ga_json(res, DEMAND, "Out.json")
        display_factory_layout("Out.json")

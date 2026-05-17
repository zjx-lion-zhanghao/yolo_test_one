import pulp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from typing import Dict, List, Tuple, Any
import itertools
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

class LunarLogistics54DayModel:
    """54-day cycle Earth-Moon logistics steady-state scheduling model (based on three fixed stations A, B, C)"""
    
    def __init__(self, lambda_weight: float = 1e6, T: int = 54):
        """
        Initialize 54-day cycle model
        
        Parameters:
        -----------
        lambda_weight : float
            Inventory deviation penalty weight coefficient
        T : int
            Cycle days, default 54 days
        """
        # Basic parameters
        self.T = T  # Cycle days (54 days)
        self.ports = ['A', 'B', 'C']  # Three fixed space stations
        self.times = list(range(self.T))  # Time set [0, 1, ..., 53]
        
        # Inventory parameters
        self.I_M0 = 0  # Moon initial inventory (tons)
        self.SA0 = 0  # Station A initial inventory (tons)
        self.SB0 = 0  # Station B initial inventory (tons)
        self.SC0 = 0  # Station C initial inventory (tons)
        self.daily_consumption = 1500  # Moon daily consumption (tons)
        self.I_target = 0  # Target inventory value (tons)
        
        # Transportation parameters
        self.transport_params = {
            # Fixed duration routes
            ('E', 'port', 'fast'): {
                'time': 2,      # Transportation time (days)
                'capacity': 125,  # Load capacity limit (tons/trip)
                'fixed_cost': 25000000,    # Fixed trip cost (USD/trip)
                'variable_cost': 800000    # Unit variable cost (USD/ton)
            },
            ('E', 'port', 'slow'): {
                'time': 6,      # Transportation time (days) - fixed 6 days
                'capacity': 2893,  # Load capacity limit (tons/trip)
                'fixed_cost': 25000000,    # Fixed trip cost (USD/trip)
                'variable_cost': 800000    # Unit variable cost (USD/ton)
            },
            ('E', 'M', 'direct'): {
                'time': 3,      # Transportation time (days)
                'capacity': 125,  # Load capacity limit (tons/trip)
                'fixed_cost': 25000000,    # Fixed trip cost (USD/trip)
                'variable_cost': 1000000   # Unit variable cost (USD/ton)
            },
            # Inter-station transfer (bidirectional, 1 day)
            ('port', 'port', 'transfer'): {
                'time': 1,      # Transportation time (days)
                'capacity': 125,  # Load capacity limit (tons/trip)
                'fixed_cost': 0,         # No fixed cost
                'variable_cost': 500000  # Unit variable cost (USD/ton)
            },
            # Station to Moon launch duration varies with 9-day rotation
            ('port', 'M', 'launch'): {
                'capacity': 125,  # Load capacity limit (tons/trip)
                'fixed_cost': 0,         # No fixed cost
                'variable_cost': 500000  # Unit variable cost (USD/ton)
            }
        }
        
        # Constraint parameters
        self.max_rockets_per_day = 10  # Earth daily maximum rocket launches
        
        # Additional: Slow elevator launch frequency limitation
        self.slow_elevator_interval = 6  # Slow elevator launch interval (days)
        self.slow_elevator_max_per_port_per_day = 1  # Each station maximum 1 slow elevator per day
        
        # Cost weight
        self.lambda_weight = lambda_weight  # Inventory deviation penalty weight
        
        # Initialize problem
        self.problem = None
        self.variables = {}
        self.results = {}
        
        print(f"Initializing 54-day cycle model (three fixed stations A, B, C)")
        print(f"Inventory deviation penalty weight λ: {lambda_weight:.2e}")
        print(f"Earth rocket limit: {self.max_rockets_per_day} rockets/day")
        print(f"Slow elevator constraint: 1 per port per day, interval {self.slow_elevator_interval} days")
        print(f"Initial inventory: Moon={self.I_M0}t, Station A={self.SA0}t, Station B={self.SB0}t, Station C={self.SC0}t")
        print(f"Station constraints: Shipment amount ≤ Station inventory")
        print(f"ABC station vehicle assumption: Unlimited vehicles, full load transport encouraged")
    
    def get_port_to_moon_time(self, t: int, port: str) -> int:
        """Get transportation time from station to Moon on day t (9-day rotation)"""
        # In 54-day cycle, 9-day rotation repeats 6 times
        cycle_index = t // 9
        
        if cycle_index % 3 == 0:
            # Days 0-8, 27-35: A=2 days, B=4 days, C=6 days
            port_times = {'A': 2, 'B': 4, 'C': 6}
        elif cycle_index % 3 == 1:
            # Days 9-17, 36-44: A=6 days, B=2 days, C=4 days
            port_times = {'A': 6, 'B': 2, 'C': 4}
        else:  # cycle_index % 3 == 2
            # Days 18-26, 45-53: A=4 days, B=6 days, C=2 days
            port_times = {'A': 4, 'B': 6, 'C': 2}
        
        return port_times.get(port, 3)
    
    def build_model(self):
        """Build 54-day cycle MILP model (based on three fixed stations)"""
        print("\nBuilding 54-day cycle Earth-Moon logistics scheduling model...")
        print("Key changes for ABC stations:")
        print("- Unlimited vehicles available")
        print("- Full load transport encouraged")
        print("- Simplified capacity constraints")
        print(f"- Slow elevator constraint: 1 per port per day, interval {self.slow_elevator_interval} days")
        
        # Create optimization problem
        self.problem = pulp.LpProblem('Lunar_Logistics_54_Day_Cycle_Fixed_Ports', pulp.LpMinimize)
        
        # 1. Define decision variables
        self._define_variables()
        
        # 2. Constraints
        self._add_constraints()
        
        # 3. Objective function
        self._define_objective()
    
    def _define_variables(self):
        """Define all decision variables"""
        print("Defining decision variables...")
        
        # Time range extended to 0-54 (including cycle endpoint)
        time_range = list(range(self.T + 1))
        
        # Cargo flow variables (continuous, tons)
        self.variables['x_EF'] = {}  # Rocket fast E->p
        self.variables['x_ES'] = {}  # Elevator slow E->p
        self.variables['x_EM'] = {}  # Earth direct E->M
        self.variables['x_pM'] = {}  # Station launch p->M
        self.variables['x_pq'] = {}  # Inter-station transfer p->q
        
        # Trip count variables (integer, trips) - simplified for ABC stations
        self.variables['k_EF'] = {}
        self.variables['k_ES'] = {}
        self.variables['k_EM'] = {}
        
        # Inventory variables (continuous, tons)
        self.variables['I_M'] = {}  # Moon inventory
        self.variables['S_A'] = {}  # Station A inventory
        self.variables['S_B'] = {}  # Station B inventory
        self.variables['S_C'] = {}  # Station C inventory
        
        # Moon inventory can be any real number (no lower bound)
        for t in time_range:
            self.variables['I_M'][t] = pulp.LpVariable(
                f'I_M_{t}', lowBound=None, cat='Continuous'
            )
            # Station inventory must be non-negative
            self.variables['S_A'][t] = pulp.LpVariable(f'S_A_{t}', lowBound=0, cat='Continuous')
            self.variables['S_B'][t] = pulp.LpVariable(f'S_B_{t}', lowBound=0, cat='Continuous')
            self.variables['S_C'][t] = pulp.LpVariable(f'S_C_{t}', lowBound=0, cat='Continuous')
        
        # Transportation variables
        for t in range(self.T):
            # Earth to stations
            for p in self.ports:
                # Rocket fast E->p
                self.variables['x_EF'][(t, p)] = pulp.LpVariable(
                    f'x_EF_{t}_{p}', lowBound=0, cat='Continuous'
                )
                self.variables['k_EF'][(t, p)] = pulp.LpVariable(
                    f'k_EF_{t}_{p}', lowBound=0, cat='Integer'
                )
                
                # Elevator slow E->p - changed to 0-1 variable, indicating launch or not
                self.variables['x_ES'][(t, p)] = pulp.LpVariable(
                    f'x_ES_{t}_{p}', lowBound=0, cat='Continuous'
                )
                self.variables['k_ES'][(t, p)] = pulp.LpVariable(
                    f'k_ES_{t}_{p}', lowBound=0, upBound=1, cat='Binary'  # Changed to Binary, indicating launch or not
                )
                
                # Station launch p->M
                self.variables['x_pM'][(t, p)] = pulp.LpVariable(
                    f'x_pM_{t}_{p}', lowBound=0, cat='Continuous'
                )
                
                # Inter-station transfer p->q
                for q in self.ports:
                    if p != q:
                        self.variables['x_pq'][(t, p, q)] = pulp.LpVariable(
                            f'x_pq_{t}_{p}_{q}', lowBound=0, cat='Continuous'
                        )
            
            # Earth direct to Moon
            self.variables['x_EM'][t] = pulp.LpVariable(
                f'x_EM_{t}', lowBound=0, cat='Continuous'
            )
            self.variables['k_EM'][t] = pulp.LpVariable(
                f'k_EM_{t}', lowBound=0, cat='Integer'
            )
    
    def _add_constraints(self):
        """Add all constraints"""
        print("Adding constraints...")
        
        M = 1e6  # Big M constant
        
        # 1. Initial inventory constraints (must be 0)
        self.problem += self.variables['I_M'][0] == self.I_M0
        self.problem += self.variables['S_A'][0] == self.SA0
        self.problem += self.variables['S_B'][0] == self.SB0
        self.problem += self.variables['S_C'][0] == self.SC0
        
        # 2. Inventory dynamics equations
        for t in range(self.T):
            next_day = t + 1
            
            # Moon inventory update
            earth_direct_arrival = 0
            if t >= 3:
                earth_direct_arrival = self.variables['x_EM'][t-3]
            
            port_launch_arrival = 0
            for p in self.ports:
                transport_time = self.get_port_to_moon_time(t, p)
                if t >= transport_time:
                    launch_time = t - transport_time
                    if launch_time >= 0:
                        port_launch_arrival += self.variables['x_pM'][(launch_time, p)]
            
            self.problem += (
                self.variables['I_M'][next_day] == 
                self.variables['I_M'][t] - self.daily_consumption + 
                earth_direct_arrival + port_launch_arrival
            )
            
            # Station inventory update
            for p in self.ports:
                fast_arrival = 0
                slow_arrival = 0
                transfer_in = 0
                
                port_launch_out = self.variables['x_pM'][(t, p)]
                transfer_out = 0
                
                # Rocket fast arrival (launched 2 days ago)
                if t >= 2:
                    fast_arrival = self.variables['x_EF'][(t-2, p)]
                
                # Elevator slow arrival (launched 6 days ago)
                if t >= 6:
                    slow_arrival = self.variables['x_ES'][(t-6, p)]
                
                # Inter-station transfer arrival (transferred 1 day ago)
                if t >= 1:
                    for q in self.ports:
                        if p != q:
                            transfer_in += self.variables['x_pq'].get((t-1, q, p), 0)
                
                # Inter-station transfer outbound (today's transfer)
                for q in self.ports:
                    if p != q:
                        transfer_out += self.variables['x_pq'].get((t, p, q), 0)
                
                # Station inventory dynamics equation
                if p == 'A':
                    self.problem += (
                        self.variables['S_A'][next_day] ==
                        self.variables['S_A'][t] + fast_arrival + slow_arrival + transfer_in
                        - port_launch_out - transfer_out
                    )
                elif p == 'B':
                    self.problem += (
                        self.variables['S_B'][next_day] ==
                        self.variables['S_B'][t] + fast_arrival + slow_arrival + transfer_in
                        - port_launch_out - transfer_out
                    )
                else:  # 'C'
                    self.problem += (
                        self.variables['S_C'][next_day] ==
                        self.variables['S_C'][t] + fast_arrival + slow_arrival + transfer_in
                        - port_launch_out - transfer_out
                    )
        
        # 3. Path capacity constraints
        for t in range(self.T):
            # Rocket fast E->p
            for p in self.ports:
                self.problem += (
                    self.variables['x_EF'][(t, p)] <= 
                    125 * self.variables['k_EF'][(t, p)]
                )
            
            # Elevator slow E->p - capacity constraints
            for p in self.ports:
                # If launched (k_ES=1), then transport volume must be at least 1 ton, maximum 2893 tons
                self.problem += (
                    self.variables['x_ES'][(t, p)] >= 
                    1 * self.variables['k_ES'][(t, p)]
                )
                self.problem += (
                    self.variables['x_ES'][(t, p)] <= 
                    2893 * self.variables['k_ES'][(t, p)]
                )
            
            # Earth direct to Moon
            self.problem += (
                self.variables['x_EM'][t] <= 
                125 * self.variables['k_EM'][t]
            )
            
        # 4. Additional: Slow elevator launch frequency constraints
        print("Adding slow elevator frequency constraints...")
        for p in self.ports:
            # Each station maximum 1 slow elevator launch per day
            for t in range(self.T):
                self.problem += (
                    self.variables['k_ES'][(t, p)] <= 
                    self.slow_elevator_max_per_port_per_day
                )
            
            # Slow elevator launch interval constraint: maximum 1 launch in any consecutive 6 days
            for t in range(self.T - self.slow_elevator_interval + 1):
                self.problem += (
                    pulp.lpSum(self.variables['k_ES'][(t+i, p)] for i in range(self.slow_elevator_interval)) <= 1
                )
            
            # Consider cycle boundary conditions: constraints for last few days and first few days of the cycle
            # For example, for 54-day cycle, we need to consider constraints between day 53 and days 0-4
            for t in range(self.T - self.slow_elevator_interval + 1, self.T):
                remaining = self.slow_elevator_interval - (self.T - t)
                constraint_sum = pulp.lpSum(self.variables['k_ES'][(t+i, p)] for i in range(self.T - t))
                constraint_sum += pulp.lpSum(self.variables['k_ES'][(i, p)] for i in range(remaining))
                self.problem += constraint_sum <= 1
        
        # 5. Strict rocket launch total constraints (Earth only)
        for t in range(self.T):
            rockets_to_ports = pulp.lpSum(
                self.variables['k_EF'][(t, p)] for p in self.ports
            )
            
            rockets_to_moon = self.variables['k_EM'][t]
            
            total_rockets = rockets_to_ports + rockets_to_moon
            self.problem += total_rockets <= self.max_rockets_per_day
        
        # 6. No demand no trip constraints
        M = 1e6  # Big M constant
        
        for t in range(self.T):
            # Rocket fast E->p
            for p in self.ports:
                self.problem += (
                    self.variables['k_EF'][(t, p)] <= 
                    M * self.variables['x_EF'][(t, p)]
                )
            
            # Elevator slow E->p
            for p in self.ports:
                self.problem += (
                    self.variables['k_ES'][(t, p)] <= 
                    M * self.variables['x_ES'][(t, p)]
                )
            
            # Earth direct to Moon
            self.problem += (
                self.variables['k_EM'][t] <= 
                M * self.variables['x_EM'][t]
            )
        
        # 7. Station shipment amount ≤ Station inventory
        print("Adding inventory shipment constraint: Station shipment amount ≤ Station inventory...")
        
        for t in range(self.T):
            # Station A
            a_launch = self.variables['x_pM'][(t, 'A')]
            a_transfer_out = 0
            for q in self.ports:
                if q != 'A':
                    a_transfer_out += self.variables['x_pq'].get((t, 'A', q), 0)
            
            # Station B
            b_launch = self.variables['x_pM'][(t, 'B')]
            b_transfer_out = 0
            for q in self.ports:
                if q != 'B':
                    b_transfer_out += self.variables['x_pq'].get((t, 'B', q), 0)
            
            # Station C
            c_launch = self.variables['x_pM'][(t, 'C')]
            c_transfer_out = 0
            for q in self.ports:
                if q != 'C':
                    c_transfer_out += self.variables['x_pq'].get((t, 'C', q), 0)
            
            # Constraint: shipment cannot exceed inventory
            self.problem += (a_launch + a_transfer_out) <= self.variables['S_A'][t]
            self.problem += (b_launch + b_transfer_out) <= self.variables['S_B'][t]
            self.problem += (c_launch + c_transfer_out) <= self.variables['S_C'][t]
        
        # 8. Cycle inventory conservation constraint (steady-state core)
        self.problem += self.variables['I_M'][self.T] == self.variables['I_M'][0]
        self.problem += self.variables['S_A'][self.T] == self.variables['S_A'][0]
        self.problem += self.variables['S_B'][self.T] == self.variables['S_B'][0]
        self.problem += self.variables['S_C'][self.T] == self.variables['S_C'][0]
    
    def _define_objective(self):
        """Define objective function (total transportation cost + λ * inventory square sum penalty)"""
        print("Defining objective function (deviation square penalty)...")
        
        total_cost = 0
        
        # 1. Transportation cost
        for t in range(self.T):
            # Earth to stations (rocket fast + elevator slow)
            for p in self.ports:
                # Fixed trip cost
                total_cost += 25000000 * (
                    self.variables['k_EF'][(t, p)] + 
                    self.variables['k_ES'][(t, p)]
                )
                # Variable cost
                total_cost += 800000 * (
                    self.variables['x_EF'][(t, p)] + 
                    self.variables['x_ES'][(t, p)]
                )
            
            # Earth direct to Moon
            total_cost += 25000000 * self.variables['k_EM'][t]
            total_cost += 1000000 * self.variables['x_EM'][t]
            
            # Station launch to Moon (no fixed cost, only variable cost)
            for p in self.ports:
                total_cost += 500000 * self.variables['x_pM'][(t, p)]
            
            # Inter-station transfer cost (no fixed cost, only variable cost)
            for p in self.ports:
                for q in self.ports:
                    if p != q:
                        total_cost += 500000 * self.variables['x_pq'][(t, p, q)]
        
        # 2. Inventory deviation penalty (absolute value approximation)
        abs_I_M = {}
        for t in range(self.T + 1):
            abs_I_M[t] = pulp.LpVariable(f'abs_I_M_{t}', lowBound=0, cat='Continuous')
            self.problem += abs_I_M[t] >= self.variables['I_M'][t]
            self.problem += abs_I_M[t] >= -self.variables['I_M'][t]
        
        deviation_penalty = pulp.lpSum(abs_I_M[t] for t in range(self.T + 1))
        
        total_objective = total_cost + self.lambda_weight * deviation_penalty
        
        self.problem += total_objective
        
        print(f"Penalty type: Absolute value approximation of square penalty (λ={self.lambda_weight:.2e})")
    
    def solve(self, time_limit: int = 18000, verbose: bool = True):  # 修改为5小时（18000秒）
        """Solve model (54-day cycle)"""
        if verbose:
            print(f"\nSolving 54-day cycle model (time limit: {time_limit} seconds = {time_limit/3600:.1f} hours)...")
            print(f"Earth rocket limit: {self.max_rockets_per_day} rockets/day")
            print(f"Slow elevator constraint: 1 per port per day, interval {self.slow_elevator_interval} days")
            print(f"Cycle length: {self.T} days")
            print(f"Inventory penalty weight: λ={self.lambda_weight:.2e}")
            print(f"ABC stations: Unlimited vehicles, full load transport encouraged")
        
        solver = pulp.PULP_CBC_CMD(msg=verbose, timeLimit=time_limit)
        
        status = self.problem.solve(solver)
        
        if pulp.LpStatus[status] == 'Optimal':
            if verbose:
                print(f"Solution successful! Objective value: {pulp.value(self.problem.objective):,.2f} USD")
            self._extract_results()
            return True
        elif pulp.LpStatus[status] == 'Feasible':
            if verbose:
                print(f"Feasible solution found! Objective value: {pulp.value(self.problem.objective):,.2f} USD")
            self._extract_results()
            return True
        else:
            if verbose:
                print(f"Solution status: {pulp.LpStatus[status]}")
            return False
    
    def _extract_results(self):
        """Extract solution results"""
        self.results = {
            'Moon Inventory': {},
            'Station Inventory': {'A': {}, 'B': {}, 'C': {}},
            'Transport Operations': {},
            'Costs': {},
            'Rocket Usage': {},
            'Station Transport Details': {'A': {}, 'B': {}, 'C': {}},
            'Inter-station Transport': {'A→B': 0, 'A→C': 0, 'B→A': 0, 'B→C': 0, 'C→A': 0, 'C→B': 0},
            'Daily Port Operations': {'A': {}, 'B': {}, 'C': {}},
            'Earth Direct Launch Schedule': {},
            'Slow Elevator Schedule': {'A': {}, 'B': {}, 'C': {}}  # New: Slow elevator launch schedule
        }
        
        # Initialize daily port operations data structure
        for port in self.ports:
            for t in range(self.T):
                self.results['Daily Port Operations'][port][t] = {
                    'Rocket Fast': {'Cargo': 0, 'Trips': 0},
                    'Elevator Slow': {'Cargo': 0, 'Trips': 0},
                    'Station Launch': {'Cargo': 0, 'Transport Time': self.get_port_to_moon_time(t, port)},
                    'Inter-station Transfer Send': {'Total': 0, 'Details': {}},
                    'Inter-station Transfer Receive': {'Total': 0, 'Details': {}},
                    'Total Input': 0,
                    'Total Output': 0,
                    'Net Change': 0
                }
                # Initialize slow elevator launch schedule
                self.results['Slow Elevator Schedule'][port][t] = {
                    'Launch': 0,  # Whether launched (0 or 1)
                    'Cargo': 0    # Transport volume (tons)
                }
        
        # Initialize Earth direct launch schedule
        for t in range(self.T):
            self.results['Earth Direct Launch Schedule'][t] = {
                'Cargo': 0,
                'Trips': 0
            }
        
        # Inventory data
        for t in range(self.T + 1):
            self.results['Moon Inventory'][t] = pulp.value(self.variables['I_M'][t])
            self.results['Station Inventory']['A'][t] = pulp.value(self.variables['S_A'][t])
            self.results['Station Inventory']['B'][t] = pulp.value(self.variables['S_B'][t])
            self.results['Station Inventory']['C'][t] = pulp.value(self.variables['S_C'][t])
        
        # Transport operations data
        transport_data = []
        
        # Station transport statistics initialization
        for p in self.ports:
            self.results['Station Transport Details'][p] = {
                'Rocket Fast Receive': 0,
                'Elevator Slow Receive': 0,
                'Station Launch Send': 0,
                'Inter-station Transfer Send': 0,
                'Inter-station Transfer Receive': 0,
                'Transport Time Distribution': {}
            }
        
        for t in range(self.T):
            # Earth to station transportation
            for p in self.ports:
                # Rocket fast
                x_ef = pulp.value(self.variables['x_EF'][(t, p)])
                k_ef = pulp.value(self.variables['k_EF'][(t, p)])
                if x_ef > 0.1 or k_ef > 0:
                    transport_data.append({
                        'Day': t,
                        'Origin': 'Earth',
                        'Destination': f'{p}',
                        'Transport Type': 'Rocket Fast',
                        'Cargo (tons)': x_ef,
                        'Trip Count': k_ef,
                        'Arrival Day': t + 2,
                        'Transport Time (days)': 2
                    })
                    self.results['Station Transport Details'][p]['Rocket Fast Receive'] += x_ef
                    self.results['Station Transport Details'][p]['Transport Time Distribution'][2] = self.results['Station Transport Details'][p]['Transport Time Distribution'].get(2, 0) + x_ef
                    
                    self.results['Daily Port Operations'][p][t]['Rocket Fast']['Cargo'] = x_ef
                    self.results['Daily Port Operations'][p][t]['Rocket Fast']['Trips'] = k_ef
                    self.results['Daily Port Operations'][p][t]['Total Input'] += x_ef
                
                # Elevator slow
                x_es = pulp.value(self.variables['x_ES'][(t, p)])
                k_es = pulp.value(self.variables['k_ES'][(t, p)])
                if x_es > 0.1 or k_es > 0:
                    transport_data.append({
                        'Day': t,
                        'Origin': 'Earth',
                        'Destination': f'{p}',
                        'Transport Type': 'Elevator Slow',
                        'Cargo (tons)': x_es,
                        'Trip Count': k_es,
                        'Arrival Day': t + 6,
                        'Transport Time (days)': 6
                    })
                    self.results['Station Transport Details'][p]['Elevator Slow Receive'] += x_es
                    self.results['Station Transport Details'][p]['Transport Time Distribution'][6] = self.results['Station Transport Details'][p]['Transport Time Distribution'].get(6, 0) + x_es
                    
                    self.results['Daily Port Operations'][p][t]['Elevator Slow']['Cargo'] = x_es
                    self.results['Daily Port Operations'][p][t]['Elevator Slow']['Trips'] = k_es
                    self.results['Daily Port Operations'][p][t]['Total Input'] += x_es
                    
                    # Record slow elevator launch schedule
                    self.results['Slow Elevator Schedule'][p][t]['Launch'] = k_es
                    self.results['Slow Elevator Schedule'][p][t]['Cargo'] = x_es
            
            # Earth direct to Moon
            x_em = pulp.value(self.variables['x_EM'][t])
            k_em = pulp.value(self.variables['k_EM'][t])
            if x_em > 0.1 or k_em > 0:
                transport_data.append({
                    'Day': t,
                    'Origin': 'Earth',
                    'Destination': 'Moon',
                    'Transport Type': 'Earth Direct',
                    'Cargo (tons)': x_em,
                    'Trip Count': k_em,
                    'Arrival Day': t + 3,
                    'Transport Time (days)': 3
                })
                self.results['Earth Direct Launch Schedule'][t]['Cargo'] = x_em
                self.results['Earth Direct Launch Schedule'][t]['Trips'] = k_em
            
            # Station launch to Moon
            for p in self.ports:
                x_pm = pulp.value(self.variables['x_pM'][(t, p)])
                transport_time = self.get_port_to_moon_time(t, p)
                if x_pm > 0.1:
                    trips = np.ceil(x_pm / 125) if x_pm > 0 else 0
                    
                    transport_data.append({
                        'Day': t,
                        'Origin': f'{p}',
                        'Destination': 'Moon',
                        'Transport Type': 'Station Launch',
                        'Cargo (tons)': x_pm,
                        'Trip Count': trips,
                        'Arrival Day': t + transport_time,
                        'Transport Time (days)': transport_time
                    })
                    self.results['Station Transport Details'][p]['Station Launch Send'] += x_pm
                    self.results['Station Transport Details'][p]['Transport Time Distribution'][transport_time] = self.results['Station Transport Details'][p]['Transport Time Distribution'].get(transport_time, 0) + x_pm
                    
                    self.results['Daily Port Operations'][p][t]['Station Launch']['Cargo'] = x_pm
                    self.results['Daily Port Operations'][p][t]['Total Output'] += x_pm
            
            # Inter-station transfer
            for p in self.ports:
                for q in self.ports:
                    if p != q:
                        x_pq = pulp.value(self.variables['x_pq'].get((t, p, q), 0))
                        if x_pq > 0.1:
                            trips = np.ceil(x_pq / 125) if x_pq > 0 else 0
                            
                            transport_data.append({
                                'Day': t,
                                'Origin': f'{p}',
                                'Destination': f'{q}',
                                'Transport Type': 'Inter-station Transfer',
                                'Cargo (tons)': x_pq,
                                'Trip Count': trips,
                                'Arrival Day': t + 1,
                                'Transport Time (days)': 1
                            })
                            self.results['Station Transport Details'][p]['Inter-station Transfer Send'] += x_pq
                            self.results['Station Transport Details'][q]['Inter-station Transfer Receive'] += x_pq
                            self.results['Inter-station Transport'][f'{p}→{q}'] += x_pq
                            
                            self.results['Daily Port Operations'][p][t]['Inter-station Transfer Send']['Total'] += x_pq
                            if q not in self.results['Daily Port Operations'][p][t]['Inter-station Transfer Send']['Details']:
                                self.results['Daily Port Operations'][p][t]['Inter-station Transfer Send']['Details'][q] = 0
                            self.results['Daily Port Operations'][p][t]['Inter-station Transfer Send']['Details'][q] += x_pq
                            self.results['Daily Port Operations'][p][t]['Total Output'] += x_pq
                            
                            self.results['Daily Port Operations'][q][t]['Inter-station Transfer Receive']['Total'] += x_pq
                            if p not in self.results['Daily Port Operations'][q][t]['Inter-station Transfer Receive']['Details']:
                                self.results['Daily Port Operations'][q][t]['Inter-station Transfer Receive']['Details'][p] = 0
                            self.results['Daily Port Operations'][q][t]['Inter-station Transfer Receive']['Details'][p] += x_pq
                            self.results['Daily Port Operations'][q][t]['Total Input'] += x_pq
        
        # Calculate net change for daily port operations
        for port in self.ports:
            for t in range(self.T):
                self.results['Daily Port Operations'][port][t]['Net Change'] = (
                    self.results['Daily Port Operations'][port][t]['Total Input'] - 
                    self.results['Daily Port Operations'][port][t]['Total Output']
                )
        
        self.results['Transport Operations'] = pd.DataFrame(transport_data)
        
        # Calculate rocket usage (Earth only)
        daily_rockets = []
        for t in range(self.T):
            daily_total = 0
            for p in self.ports:
                daily_total += pulp.value(self.variables['k_EF'][(t, p)])
            daily_total += pulp.value(self.variables['k_EM'][t])
            daily_rockets.append(daily_total)
        
        self.results['Rocket Usage'] = {
            'Daily Rockets': daily_rockets,
            'Total Rockets': sum(daily_rockets),
            'Average Daily Rockets': sum(daily_rockets) / self.T,
            'Max Daily Rockets': max(daily_rockets) if daily_rockets else 0,
            'Min Daily Rockets': min(daily_rockets) if daily_rockets else 0,
            'Rocket Usage Rate': sum(daily_rockets) / (self.T * self.max_rockets_per_day) * 100
        }
        
        # Calculate costs
        self._calculate_costs()
        
        # Calculate inventory statistics
        lunar_inventory = [self.results['Moon Inventory'][t] for t in range(self.T + 1)]
        self.results['Inventory Statistics'] = {
            'Moon Inventory Mean': np.mean(lunar_inventory),
            'Moon Inventory Std': np.std(lunar_inventory),
            'Moon Inventory Max': max(lunar_inventory),
            'Moon Inventory Min': min(lunar_inventory),
            'Moon Inventory Square Sum': sum(i**2 for i in lunar_inventory)
        }
        
        # New: Slow elevator launch statistics
        self._calculate_slow_elevator_stats()
        
        # Calculate additional cost analysis
        self._calculate_additional_costs()
    
    def _calculate_slow_elevator_stats(self):
        """Calculate slow elevator launch statistics"""
        slow_elevator_stats = {}
        
        for port in self.ports:
            launch_days = []
            launch_cargos = []
            
            for t in range(self.T):
                if self.results['Slow Elevator Schedule'][port][t]['Launch'] > 0.5:  # Slow elevator launched
                    launch_days.append(t)
                    launch_cargos.append(self.results['Slow Elevator Schedule'][port][t]['Cargo'])
            
            if launch_days:
                intervals = [launch_days[i+1] - launch_days[i] for i in range(len(launch_days)-1)]
                
                slow_elevator_stats[port] = {
                    'Total Launches': len(launch_days),
                    'Launch Days': launch_days,
                    'Total Cargo': sum(launch_cargos),
                    'Average Cargo per Launch': np.mean(launch_cargos) if launch_cargos else 0,
                    'Min Interval': min(intervals) if intervals else 0,
                    'Max Interval': max(intervals) if intervals else 0,
                    'Average Interval': np.mean(intervals) if intervals else 0
                }
            else:
                slow_elevator_stats[port] = {
                    'Total Launches': 0,
                    'Launch Days': [],
                    'Total Cargo': 0,
                    'Average Cargo per Launch': 0,
                    'Min Interval': 0,
                    'Max Interval': 0,
                    'Average Interval': 0
                }
        
        self.results['Slow Elevator Stats'] = slow_elevator_stats
    
    def _calculate_costs(self):
        """Calculate various costs"""
        total_cost = 0
        fixed_cost = 0
        variable_cost = 0
        
        # Transportation cost
        for t in range(self.T):
            # Fixed trip cost
            for p in self.ports:
                k_ef = pulp.value(self.variables['k_EF'][(t, p)])
                k_es = pulp.value(self.variables['k_ES'][(t, p)])
                fixed_cost += 25000000 * (k_ef + k_es)
            
            k_em = pulp.value(self.variables['k_EM'][t])
            fixed_cost += 25000000 * k_em
            
            # Variable cost
            for p in self.ports:
                x_ef = pulp.value(self.variables['x_EF'][(t, p)])
                x_es = pulp.value(self.variables['x_ES'][(t, p)])
                variable_cost += 800000 * (x_ef + x_es)
            
            x_em = pulp.value(self.variables['x_EM'][t])
            variable_cost += 1000000 * x_em
            
            # Station launch to Moon (no fixed cost, only variable cost)
            for p in self.ports:
                x_pm = pulp.value(self.variables['x_pM'][(t, p)])
                variable_cost += 500000 * x_pm
            
            # Inter-station transfer cost (no fixed cost, only variable cost)
            for p in self.ports:
                for q in self.ports:
                    if p != q:
                        x_pq = pulp.value(self.variables['x_pq'].get((t, p, q), 0))
                        variable_cost += 500000 * x_pq
        
        # Inventory deviation penalty (absolute value approximation)
        deviation_penalty = 0
        for t in range(self.T + 1):
            i_m = pulp.value(self.variables['I_M'][t])
            deviation_penalty += abs(i_m)
        
        deviation_cost = self.lambda_weight * deviation_penalty
        
        total_cost = fixed_cost + variable_cost + deviation_cost
        
        self.results['Costs'] = {
            'Total Cost (USD)': total_cost,
            'Fixed Trip Cost (USD)': fixed_cost,
            'Variable Transport Cost (USD)': variable_cost,
            'Inventory Deviation Penalty (USD)': deviation_cost,
            'Daily Average Cost (USD)': total_cost / self.T,
            'Cycle Initial Inventory (tons)': pulp.value(self.variables['I_M'][0]),
            'Cycle Final Inventory (tons)': pulp.value(self.variables['I_M'][self.T]),
            'Inventory Square Sum Penalty': sum(pulp.value(self.variables['I_M'][t])**2 for t in range(self.T + 1)),
            'Penalty Weight λ': self.lambda_weight,
            'Total Inventory Deviation (tons)': deviation_penalty
        }
    
    def _calculate_additional_costs(self):
        """Calculate additional cost analysis"""
        transport_mode_costs = {
            'Rocket Fast': 0,
            'Elevator Slow': 0,
            'Earth Direct': 0,
            'Station Launch': 0,
            'Inter-station Transfer': 0
        }
        
        port_costs = {p: 0 for p in self.ports}
        
        transport_mode_cost_breakdown = {
            'Rocket Fast': {'Fixed Cost': 0, 'Variable Cost': 0, 'Total Cost': 0},
            'Elevator Slow': {'Fixed Cost': 0, 'Variable Cost': 0, 'Total Cost': 0},
            'Earth Direct': {'Fixed Cost': 0, 'Variable Cost': 0, 'Total Cost': 0},
            'Station Launch': {'Fixed Cost': 0, 'Variable Cost': 0, 'Total Cost': 0},
            'Inter-station Transfer': {'Fixed Cost': 0, 'Variable Cost': 0, 'Total Cost': 0}
        }
        
        for t in range(self.T):
            for p in self.ports:
                # Rocket fast cost
                x_ef = pulp.value(self.variables['x_EF'][(t, p)])
                k_ef = pulp.value(self.variables['k_EF'][(t, p)])
                if x_ef > 0.1 or k_ef > 0:
                    fixed_cost_part = 25000000 * k_ef
                    variable_cost_part = 800000 * x_ef
                    cost = fixed_cost_part + variable_cost_part
                    transport_mode_costs['Rocket Fast'] += cost
                    transport_mode_cost_breakdown['Rocket Fast']['Fixed Cost'] += fixed_cost_part
                    transport_mode_cost_breakdown['Rocket Fast']['Variable Cost'] += variable_cost_part
                    transport_mode_cost_breakdown['Rocket Fast']['Total Cost'] += cost
                    port_costs[p] += cost
                
                # Elevator slow cost
                x_es = pulp.value(self.variables['x_ES'][(t, p)])
                k_es = pulp.value(self.variables['k_ES'][(t, p)])
                if x_es > 0.1 or k_es > 0:
                    fixed_cost_part = 25000000 * k_es
                    variable_cost_part = 800000 * x_es
                    cost = fixed_cost_part + variable_cost_part
                    transport_mode_costs['Elevator Slow'] += cost
                    transport_mode_cost_breakdown['Elevator Slow']['Fixed Cost'] += fixed_cost_part
                    transport_mode_cost_breakdown['Elevator Slow']['Variable Cost'] += variable_cost_part
                    transport_mode_cost_breakdown['Elevator Slow']['Total Cost'] += cost
                    port_costs[p] += cost
            
            # Earth direct cost
            x_em = pulp.value(self.variables['x_EM'][t])
            k_em = pulp.value(self.variables['k_EM'][t])
            if x_em > 0.1 or k_em > 0:
                fixed_cost_part = 25000000 * k_em
                variable_cost_part = 1000000 * x_em
                cost = fixed_cost_part + variable_cost_part
                transport_mode_costs['Earth Direct'] += cost
                transport_mode_cost_breakdown['Earth Direct']['Fixed Cost'] += fixed_cost_part
                transport_mode_cost_breakdown['Earth Direct']['Variable Cost'] += variable_cost_part
                transport_mode_cost_breakdown['Earth Direct']['Total Cost'] += cost
            
            # Station launch cost
            for p in self.ports:
                x_pm = pulp.value(self.variables['x_pM'][(t, p)])
                if x_pm > 0.1:
                    fixed_cost_part = 0
                    variable_cost_part = 500000 * x_pm
                    cost = fixed_cost_part + variable_cost_part
                    transport_mode_costs['Station Launch'] += cost
                    transport_mode_cost_breakdown['Station Launch']['Fixed Cost'] += fixed_cost_part
                    transport_mode_cost_breakdown['Station Launch']['Variable Cost'] += variable_cost_part
                    transport_mode_cost_breakdown['Station Launch']['Total Cost'] += cost
                    port_costs[p] += cost
            
            # Inter-station transfer cost
            for p in self.ports:
                for q in self.ports:
                    if p != q:
                        x_pq = pulp.value(self.variables['x_pq'].get((t, p, q), 0))
                        if x_pq > 0.1:
                            fixed_cost_part = 0
                            variable_cost_part = 500000 * x_pq
                            cost = fixed_cost_part + variable_cost_part
                            transport_mode_costs['Inter-station Transfer'] += cost
                            transport_mode_cost_breakdown['Inter-station Transfer']['Fixed Cost'] += fixed_cost_part
                            transport_mode_cost_breakdown['Inter-station Transfer']['Variable Cost'] += variable_cost_part
                            transport_mode_cost_breakdown['Inter-station Transfer']['Total Cost'] += cost
                            port_costs[p] += cost
        
        # Calculate cost efficiency metrics
        total_transport = 0
        if len(self.results['Transport Operations']) > 0:
            total_transport = self.results['Transport Operations']['Cargo (tons)'].sum()
        
        cost_per_ton = self.results['Costs']['Total Cost (USD)'] / total_transport if total_transport > 0 else 0
        
        self.results['Additional Cost Analysis'] = {
            'Transport Mode Cost Distribution': transport_mode_costs,
            'Transport Mode Cost Breakdown': transport_mode_cost_breakdown,
            'Station Transport Costs': port_costs,
            'Cost per Ton (USD/ton)': cost_per_ton,
            'Transport Efficiency Metrics': {
                'Total Transport (tons)': total_transport,
                'Total Cost (USD)': self.results['Costs']['Total Cost (USD)'],
                'Cost Efficiency Ratio': cost_per_ton
            }
        }
    
    # ==================== IMPROVED VISUALIZATION METHODS ====================
    
    def plot_inventory_levels(self):
        """Improved: Plot inventory level changes for Moon and three stations in one figure"""
        print("\n" + "="*80)
        print("Plotting Inventory Level Changes (Combined)")
        print("="*80)
        
        # Create figure with improved layout
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid for main plot and statistics
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)
        
        # Main plot area
        ax_main = fig.add_subplot(gs[0])
        
        days = list(range(self.T + 1))
        
        # Get inventory data
        moon_inventory = [self.results['Moon Inventory'][t] for t in days]
        station_a_inventory = [self.results['Station Inventory']['A'][t] for t in days]
        station_b_inventory = [self.results['Station Inventory']['B'][t] for t in days]
        station_c_inventory = [self.results['Station Inventory']['C'][t] for t in days]
        
        # Define colors with better contrast
        colors = {
            'Moon': '#1f77b4',      # Blue
            'Station A': '#2ca02c',  # Green
            'Station B': '#d62728',  # Red
            'Station C': '#9467bd'   # Purple
        }
        
        # Plot all inventories on one chart with different styles
        line_moon = ax_main.plot(days, moon_inventory, color=colors['Moon'], 
                                linewidth=2.5, marker='o', markersize=4, 
                                label='Moon Inventory', zorder=5)
        
        line_a = ax_main.plot(days, station_a_inventory, color=colors['Station A'], 
                             linewidth=2, linestyle='-', marker='s', markersize=4, 
                             label='Station A', zorder=4)
        
        line_b = ax_main.plot(days, station_b_inventory, color=colors['Station B'], 
                             linewidth=2, linestyle='--', marker='^', markersize=4, 
                             label='Station B', zorder=3)
        
        line_c = ax_main.plot(days, station_c_inventory, color=colors['Station C'], 
                             linewidth=2, linestyle='-.', marker='d', markersize=4, 
                             label='Station C', zorder=2)
        
        # Add reference lines
        ax_main.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax_main.axhline(y=-self.daily_consumption, color='orange', 
                       linestyle=':', linewidth=1, alpha=0.5)
        
        # Add fill between for Moon inventory (positive vs negative)
        ax_main.fill_between(days, 0, moon_inventory, 
                            where=(np.array(moon_inventory) >= 0), 
                            alpha=0.15, color=colors['Moon'], label='Moon Positive')
        ax_main.fill_between(days, 0, moon_inventory, 
                            where=(np.array(moon_inventory) < 0), 
                            alpha=0.15, color='red', label='Moon Negative')
        
        # Set labels and title
        ax_main.set_xlabel('Day', fontsize=12, fontweight='medium')
        ax_main.set_ylabel('Inventory (tons)', fontsize=12, fontweight='medium')
        ax_main.set_title('Inventory Levels - Moon and Stations A, B, C (54-day Cycle)', 
                         fontsize=14, fontweight='bold', pad=15)
        
        # Improve legend
        ax_main.legend(loc='upper left', fontsize=10, framealpha=0.9, 
                      edgecolor='gray', fancybox=True)
        
        # Set grid with improved styling
        ax_main.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Set tick parameters
        ax_main.tick_params(axis='both', labelsize=10)
        ax_main.set_xlim(0, self.T)
        
        # Add light vertical grid lines for 9-day intervals (less obtrusive)
        for i in range(0, self.T, 9):
            ax_main.axvline(x=i, color='lightgray', linestyle=':', 
                          linewidth=0.5, alpha=0.5, zorder=1)
        
        # Statistics panel (right side)
        ax_stats = fig.add_subplot(gs[1])
        ax_stats.axis('off')  # Hide axes
        
        # Prepare statistics text
        stats_text = "Inventory Statistics\n\n"
        stats_text += f"Moon:\n"
        stats_text += f"  Mean: {self.results['Inventory Statistics']['Moon Inventory Mean']:.0f} t\n"
        stats_text += f"  Std: {self.results['Inventory Statistics']['Moon Inventory Std']:.0f} t\n"
        stats_text += f"  Max: {self.results['Inventory Statistics']['Moon Inventory Max']:.0f} t\n"
        stats_text += f"  Min: {self.results['Inventory Statistics']['Moon Inventory Min']:.0f} t\n\n"
        
        for port, data, color in zip(['A', 'B', 'C'], 
                                    [station_a_inventory, station_b_inventory, station_c_inventory],
                                    [colors['Station A'], colors['Station B'], colors['Station C']]):
            stats_text += f"Station {port}:\n"
            stats_text += f"  Max: {max(data):.0f} t\n"
            stats_text += f"  Min: {min(data):.0f} t\n"
            stats_text += f"  Mean: {np.mean(data):.0f} t\n\n"
        
        # Add consumption info
        stats_text += f"Moon Daily Consumption:\n"
        stats_text += f"  {self.daily_consumption} tons/day\n"
        
        # Add model parameters
        stats_text += f"\nModel Parameters:\n"
        stats_text += f"  Cycle: {self.T} days\n"
        stats_text += f"  λ: {self.lambda_weight:.1e}\n"
        stats_text += f"  Rocket Limit: {self.max_rockets_per_day}/day"
        
        # Display statistics as text box
        ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='whitesmoke', 
                              edgecolor='lightgray', alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('inventory_levels_combined.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Combined inventory chart saved as: inventory_levels_combined.png")
        
        # Return data for further analysis
        inventory_data = {
            'Moon': moon_inventory,
            'Station A': station_a_inventory,
            'Station B': station_b_inventory,
            'Station C': station_c_inventory,
            'Days': days
        }
        
        return inventory_data
    
    def plot_station_operations(self):
        """Improved: Plot launch plans and inventory changes for each station with better layout"""
        print("\n" + "="*80)
        print("Plotting Station Operations (Improved Layout)")
        print("="*80)
        
        # Prepare data for each station
        stations_data = {}
        
        for port in self.ports:
            daily_ops = self.results['Daily Port Operations'][port]
            
            # Prepare data arrays
            days = list(range(self.T))
            rocket_fast = [daily_ops[t]['Rocket Fast']['Cargo'] for t in days]
            elevator_slow = [daily_ops[t]['Elevator Slow']['Cargo'] for t in days]
            station_launch = [daily_ops[t]['Station Launch']['Cargo'] for t in days]
            transfer_in = [daily_ops[t]['Inter-station Transfer Receive']['Total'] for t in days]
            transfer_out = [daily_ops[t]['Inter-station Transfer Send']['Total'] for t in days]
            net_change = [daily_ops[t]['Net Change'] for t in days]
            
            inventory = [self.results['Station Inventory'][port][t] for t in range(self.T)]
            
            stations_data[port] = {
                'days': days,
                'rocket_fast': rocket_fast,
                'elevator_slow': elevator_slow,
                'station_launch': station_launch,
                'transfer_in': transfer_in,
                'transfer_out': transfer_out,
                'net_change': net_change,
                'inventory': inventory
            }
        
        # Create figure with improved layout
        fig, axes = plt.subplots(3, 2, figsize=(18, 14))
        fig.suptitle('Station Operations Analysis (54-day Cycle)', fontsize=16, fontweight='bold', y=0.98)
        
        # Define improved color scheme
        colors = {
            'rocket_fast': '#FF6B6B',  # Coral red
            'elevator_slow': '#4ECDC4',  # Turquoise
            'station_launch': '#45B7D1',  # Sky blue
            'transfer_in': '#96CEB4',  # Sage green
            'transfer_out': '#FFD166',  # Yellow
        }
        
        station_names = {'A': 'Station A', 'B': 'Station B', 'C': 'Station C'}
        
        for idx, port in enumerate(self.ports):
            data = stations_data[port]
            days = data['days']
            
            # 1. Input operations (stacked bar) - LEFT SIDE
            ax1 = axes[idx, 0]
            
            # Stack input components
            bottom = np.zeros(len(days))
            input_components = [
                ('Rocket Fast', data['rocket_fast'], colors['rocket_fast']),
                ('Elevator Slow', data['elevator_slow'], colors['elevator_slow']),
                ('Transfer In', data['transfer_in'], colors['transfer_in'])
            ]
            
            bars = []
            bar_labels = []
            for label, values, color in input_components:
                if max(values) > 0:  # Only plot if there's data
                    bar = ax1.bar(days, values, bottom=bottom, label=label, 
                                  color=color, alpha=0.8, width=0.8)
                    bars.append(bar)
                    bar_labels.append(label)
                    bottom += values
            
            ax1.set_xlabel('Day', fontsize=11)
            ax1.set_ylabel('Input Volume (tons)', fontsize=11)
            ax1.set_title(f'{station_names[port]} - Input Operations', 
                         fontsize=13, fontweight='bold', pad=10)
            
            # Improve legend placement
            if idx == 0:  # Only show legend on first row
                ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
            
            ax1.grid(True, alpha=0.2, axis='y')
            ax1.tick_params(axis='both', labelsize=9)
            ax1.set_xlim(-0.5, self.T-0.5)
            
            # Add total input as annotation (clean placement)
            total_input = sum(data['rocket_fast']) + sum(data['elevator_slow']) + sum(data['transfer_in'])
            ax1.text(0.02, 0.95, f'Total: {total_input:.0f} t', 
                    transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            
            # 2. Output operations and inventory - RIGHT SIDE
            ax2 = axes[idx, 1]
            ax2_inv = ax2.twinx()
            
            # Plot output operations as stacked bars
            output_width = 0.6
            launch_bars = ax2.bar(days, data['station_launch'], width=output_width, 
                                 color=colors['station_launch'], alpha=0.8, label='Launch to Moon')
            
            transfer_bars = ax2.bar(days, data['transfer_out'], width=output_width, 
                                   bottom=data['station_launch'], 
                                   color=colors['transfer_out'], alpha=0.8, label='Transfer Out')
            
            # Plot inventory as line on secondary axis
            inv_line = ax2_inv.plot(days, data['inventory'], color='#333333', 
                                   linewidth=2.5, marker='o', markersize=3, 
                                   label='Inventory', zorder=5)
            
            ax2.set_xlabel('Day', fontsize=11)
            ax2.set_ylabel('Output Volume (tons)', fontsize=11, 
                          color=colors['station_launch'])
            ax2_inv.set_ylabel('Inventory (tons)', fontsize=11, color='#333333')
            ax2.set_title(f'{station_names[port]} - Output & Inventory', 
                         fontsize=13, fontweight='bold', pad=10)
            
            # Color the y-axis labels to match data
            ax2.tick_params(axis='y', labelcolor=colors['station_launch'])
            ax2_inv.tick_params(axis='y', labelcolor='#333333')
            
            # Create combined legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_inv.get_legend_handles_labels()
            
            if idx == 0:  # Only show legend on first row
                ax2.legend(lines1 + lines2, labels1 + labels2, 
                          fontsize=9, loc='upper left', framealpha=0.9)
            
            ax2.grid(True, alpha=0.2, axis='y')
            ax2.tick_params(axis='both', labelsize=9)
            ax2_inv.tick_params(axis='y', labelsize=9)
            ax2.set_xlim(-0.5, self.T-0.5)
            
            # Add statistics with clean layout
            total_output = sum(data['station_launch']) + sum(data['transfer_out'])
            max_inventory = max(data['inventory'])
            min_inventory = min(data['inventory'])
            mean_inventory = np.mean(data['inventory'])
            
            stats_text = f"""Output Stats:
Total: {total_output:.0f} t
Inv Max: {max_inventory:.0f} t
Inv Min: {min_inventory:.0f} t
Inv Mean: {mean_inventory:.0f} t"""
            
            # Place stats in upper right corner
            ax2.text(0.98, 0.95, stats_text, transform=ax2.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add light vertical grid lines for 9-day intervals (all subplots)
        for idx in range(3):
            for col in range(2):
                for i in range(0, self.T, 9):
                    axes[idx, col].axvline(x=i, color='lightgray', 
                                          linestyle=':', linewidth=0.5, alpha=0.3, zorder=1)
        
        plt.tight_layout()
        plt.savefig('station_operations_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Improved station operations chart saved as: station_operations_improved.png")
        
        return stations_data
    
    def plot_slow_elevator_schedule(self):
        """Improved: Plot slow elevator launch schedule in a cleaner layout"""
        print("\n" + "="*80)
        print("Plotting Slow Elevator Launch Schedule (Improved)")
        print("="*80)
        
        # Create figure with improved layout
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Slow Elevator Launch Schedule (54-day Cycle)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        station_names = {'A': 'Station A', 'B': 'Station B', 'C': 'Station C'}
        colors = {'A': '#2E8B57', 'B': '#DC143C', 'C': '#6A5ACD'}  # SeaGreen, Crimson, SlateBlue
        
        days = list(range(self.T))
        
        for idx, port in enumerate(self.ports):
            ax = axes[idx]
            
            # Get slow elevator launch data
            launch_days = []
            launch_cargos = []
            
            for t in days:
                if self.results['Slow Elevator Schedule'][port][t]['Launch'] > 0.5:
                    launch_days.append(t)
                    launch_cargos.append(self.results['Slow Elevator Schedule'][port][t]['Cargo'])
            
            # Plot launch points with improved styling
            if launch_days:
                # Plot scatter points with consistent size
                scatter = ax.scatter(launch_days, launch_cargos, s=120, 
                                    color=colors[port], edgecolors='white', 
                                    linewidth=1.5, zorder=5, alpha=0.9,
                                    label=f'Slow Elevator Launch')
                
                # Connect launch points with subtle lines
                for i in range(len(launch_days)-1):
                    ax.plot([launch_days[i], launch_days[i+1]], 
                           [launch_cargos[i], launch_cargos[i+1]], 
                           '--', color=colors[port], alpha=0.4, linewidth=1.2, zorder=4)
            
            # Draw horizontal lines for capacity limits
            ax.axhline(y=2893, color='gray', linestyle=':', linewidth=1.5, 
                      alpha=0.5, label=f'Max Capacity (2893 t)')
            ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Set labels and title
            ax.set_xlabel('Day', fontsize=11)
            ax.set_ylabel('Cargo (tons)', fontsize=11)
            ax.set_title(f'{station_names[port]}', fontsize=13, fontweight='bold', pad=10)
            
            # Add grid
            ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
            
            # Set limits
            ax.set_xlim(-1, self.T)
            ax.set_ylim(-50, 3100)  # Add some padding
            
            # Set tick parameters
            ax.tick_params(axis='both', labelsize=9)
            
            # Add light vertical grid lines at 9-day intervals
            for i in range(0, self.T, 9):
                ax.axvline(x=i, color='lightgray', linestyle=':', 
                          linewidth=0.5, alpha=0.3, zorder=1)
            
            # Add statistics box
            stats = self.results['Slow Elevator Stats'][port]
            if stats['Total Launches'] > 0:
                stats_text = f"""Launches: {stats['Total Launches']}
Total: {stats['Total Cargo']:.0f} t
Avg: {stats['Average Cargo per Launch']:.0f} t/launch
Avg Interval: {stats['Average Interval']:.1f} days"""
            else:
                stats_text = "No launches"
            
            # Place stats in upper left corner
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.9, edgecolor='lightgray'))
            
            # Add legend (only for first subplot to avoid clutter)
            if idx == 0 and launch_days:
                ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
        
        plt.tight_layout()
        plt.savefig('slow_elevator_schedule_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Improved slow elevator schedule saved as: slow_elevator_schedule_improved.png")
        
        # Print concise summary
        print("\nSlow Elevator Launch Summary:")
        print("-" * 50)
        
        for port in self.ports:
            stats = self.results['Slow Elevator Stats'][port]
            if stats['Total Launches'] > 0:
                print(f"\n{station_names[port]}:")
                print(f"  Launches: {stats['Total Launches']}")
                print(f"  Total Cargo: {stats['Total Cargo']:.0f} t")
                print(f"  Launch Days: {stats['Launch Days']}")
        
        return self.results['Slow Elevator Stats']
    
    def plot_launch_schedule_heatmap(self):
        """Improved: Plot launch schedule heatmap without cycle markers"""
        print("\n" + "="*80)
        print("Plotting Launch Schedule Heatmap (Clean Version)")
        print("="*80)
        
        # Prepare data
        days = list(range(self.T))
        transport_types = [
            'Earth→A(Rocket Fast)', 'Earth→B(Rocket Fast)', 'Earth→C(Rocket Fast)',
            'Earth→A(Elevator Slow)', 'Earth→B(Elevator Slow)', 'Earth→C(Elevator Slow)',
            'Earth→Moon(Direct)',
            'A→Moon', 'B→Moon', 'C→Moon',
            'A→B(Transfer)', 'A→C(Transfer)', 'B→A(Transfer)', 'B→C(Transfer)', 'C→A(Transfer)', 'C→B(Transfer)'
        ]
        
        # Create data matrix
        data_matrix = np.zeros((len(transport_types), len(days)))
        
        transport_df = self.results['Transport Operations']
        
        for i, transport_type in enumerate(transport_types):
            for j, day in enumerate(days):
                # Extract data based on transport type
                if transport_type.startswith('Earth→') and '(Rocket Fast)' in transport_type:
                    port = transport_type.split('→')[1].split('(')[0]
                    filtered = transport_df[
                        (transport_df['Day'] == day) & 
                        (transport_df['Transport Type'] == 'Rocket Fast') &
                        (transport_df['Destination'] == port)
                    ]
                elif transport_type.startswith('Earth→') and '(Elevator Slow)' in transport_type:
                    port = transport_type.split('→')[1].split('(')[0]
                    filtered = transport_df[
                        (transport_df['Day'] == day) & 
                        (transport_df['Transport Type'] == 'Elevator Slow') &
                        (transport_df['Destination'] == port)
                    ]
                elif transport_type == 'Earth→Moon(Direct)':
                    filtered = transport_df[
                        (transport_df['Day'] == day) & 
                        (transport_df['Transport Type'] == 'Earth Direct')
                    ]
                elif transport_type.endswith('→Moon'):
                    port = transport_type.split('→')[0]
                    filtered = transport_df[
                        (transport_df['Day'] == day) & 
                        (transport_df['Transport Type'] == 'Station Launch') &
                        (transport_df['Origin'] == port)
                    ]
                elif 'Transfer' in transport_type:
                    src, dst = transport_type.split('→')
                    dst = dst.split('(')[0]
                    filtered = transport_df[
                        (transport_df['Day'] == day) & 
                        (transport_df['Transport Type'] == 'Inter-station Transfer') &
                        (transport_df['Origin'] == src) &
                        (transport_df['Destination'] == dst)
                    ]
                else:
                    filtered = pd.DataFrame()
                
                if len(filtered) > 0:
                    data_matrix[i, j] = filtered['Cargo (tons)'].sum()
        
        # Create heatmap with improved styling
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # Use viridis colormap for better visual appeal
        cmap = plt.cm.viridis
        im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, 
                      interpolation='nearest', vmin=0)
        
        # Set axes with cleaner labels
        ax.set_xticks(range(0, len(days), 5))
        ax.set_xticklabels([f'D{d}' for d in range(0, len(days), 5)], 
                          fontsize=9, rotation=0)
        
        ax.set_yticks(range(len(transport_types)))
        ax.set_yticklabels(transport_types, fontsize=9)
        
        # Add color bar with label
        cbar = plt.colorbar(im, ax=ax, pad=0.01)
        cbar.set_label('Transport Volume (tons)', fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        
        # Add subtle grid (only horizontal)
        ax.set_xticks(np.arange(-0.5, len(days), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(transport_types), 1), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', 
               linewidth=0.3, alpha=0.5)
        
        # Title with better formatting
        ax.set_title('54-day Cycle Launch Schedule Heatmap', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Day', fontsize=11)
        
        # Remove cycle markers as requested
        # Add subtle vertical lines for every 9 days (less obtrusive)
        for i in range(0, self.T, 9):
            ax.axvline(x=i-0.5, color='lightgray', 
                      linestyle=':', linewidth=0.8, alpha=0.4)
        
        plt.tight_layout()
        plt.savefig('launch_schedule_heatmap_clean.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Clean heatmap saved as: launch_schedule_heatmap_clean.png")
        
        # Print summary statistics
        print("\nTransport Summary by Type:")
        print("-" * 40)
        
        # Calculate totals by transport type
        for i, transport_type in enumerate(transport_types):
            total = np.sum(data_matrix[i, :])
            if total > 0:
                print(f"{transport_type:<30} {total:>8.0f} t")
        
        return data_matrix
    
    def generate_earth_direct_schedule_table(self):
        """Generate Earth direct launch schedule timetable (table format)"""
        print("\n" + "="*80)
        print("Earth Direct Launch Schedule Timetable")
        print("="*80)
        
        schedule_data = []
        
        for t in range(self.T):
            daily_direct = self.results['Earth Direct Launch Schedule'][t]
            
            if daily_direct['Cargo'] > 0:
                schedule_data.append({
                    'Launch Day': t,
                    'Cargo (tons)': daily_direct['Cargo'],
                    'Launch Count': int(daily_direct['Trips']),
                    'Estimated Arrival Day': t + 3,
                    'Transport Time (days)': 3,
                    'Load Factor (%)': (daily_direct['Cargo'] / (125 * daily_direct['Trips']) * 100) if daily_direct['Trips'] > 0 else 0
                })
        
        if schedule_data:
            df = pd.DataFrame(schedule_data)
            
            # Calculate statistics
            total_cargo = df['Cargo (tons)'].sum()
            total_flights = df['Launch Count'].sum()
            avg_load_per_flight = total_cargo / total_flights if total_flights > 0 else 0
            avg_load_rate = df['Load Factor (%)'].mean()
            
            print(f"Earth Direct Launch Schedule Summary:")
            print(f"  Total Transport: {total_cargo:.0f} tons")
            print(f"  Total Launches: {total_flights:.0f}")
            print(f"  Average Load per Flight: {avg_load_per_flight:.1f} tons")
            print(f"  Average Load Factor: {avg_load_rate:.1f}%")
            print(f"  Launch Days: {len(df)} days (out of {self.T} days)")
            print()
            
            # Display detailed table
            print("Detailed Launch Schedule:")
            print(df.round(1).to_string(index=False))
            
            # Save as CSV file
            csv_filename = 'earth_direct_schedule.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"\nEarth direct launch schedule saved as: {csv_filename}")
            
            return df
        else:
            print("No Earth direct launch records")
            return pd.DataFrame()
    
    def generate_detailed_schedule_table(self):
        """Generate detailed launch schedule timetable (including all transport types)"""
        print("\n" + "="*80)
        print("Detailed Launch Schedule Timetable (All Transport Types)")
        print("="*80)
        
        # Extract all records from transport operations data
        transport_df = self.results['Transport Operations']
        
        if len(transport_df) == 0:
            print("No transport records")
            return pd.DataFrame()
        
        # Classify by transport type
        transport_types = transport_df['Transport Type'].unique()
        
        # Create table summarized by transport type
        detailed_schedule = []
        
        for t in range(self.T):
            day_operations = {}
            
            # Earth to station transport
            for port in self.ports:
                # Rocket fast
                rocket_fast = transport_df[
                    (transport_df['Day'] == t) & 
                    (transport_df['Transport Type'] == 'Rocket Fast') &
                    (transport_df['Destination'] == port)
                ]
                if len(rocket_fast) > 0:
                    day_operations[f'Earth→{port}(Rocket Fast)'] = {
                        'Cargo': rocket_fast['Cargo (tons)'].sum(),
                        'Launch Count': int(rocket_fast['Trip Count'].sum())
                    }
                
                # Elevator slow
                elevator_slow = transport_df[
                    (transport_df['Day'] == t) & 
                    (transport_df['Transport Type'] == 'Elevator Slow') &
                    (transport_df['Destination'] == port)
                ]
                if len(elevator_slow) > 0:
                    day_operations[f'Earth→{port}(Elevator Slow)'] = {
                        'Cargo': elevator_slow['Cargo (tons)'].sum(),
                        'Launch Count': int(elevator_slow['Trip Count'].sum())
                    }
            
            # Earth direct to Moon
            earth_direct = transport_df[
                (transport_df['Day'] == t) & 
                (transport_df['Transport Type'] == 'Earth Direct')
            ]
            if len(earth_direct) > 0:
                day_operations['Earth→Moon(Direct)'] = {
                    'Cargo': earth_direct['Cargo (tons)'].sum(),
                    'Launch Count': int(earth_direct['Trip Count'].sum())
                }
            
            # Station to Moon transport
            for port in self.ports:
                station_launch = transport_df[
                    (transport_df['Day'] == t) & 
                    (transport_df['Transport Type'] == 'Station Launch') &
                    (transport_df['Origin'] == port)
                ]
                if len(station_launch) > 0:
                    transport_time = self.get_port_to_moon_time(t, port)
                    day_operations[f'{port}→Moon(Launch)'] = {
                        'Cargo': station_launch['Cargo (tons)'].sum(),
                        'Launch Count': int(station_launch['Trip Count'].sum()),
                        'Transport Time': transport_time
                    }
            
            # Inter-station transfer
            for src in self.ports:
                for dst in self.ports:
                    if src != dst:
                        transfer = transport_df[
                            (transport_df['Day'] == t) & 
                            (transport_df['Transport Type'] == 'Inter-station Transfer') &
                            (transport_df['Origin'] == src) &
                            (transport_df['Destination'] == dst)
                        ]
                        if len(transfer) > 0:
                            day_operations[f'{src}→{dst}(Transfer)'] = {
                                'Cargo': transfer['Cargo (tons)'].sum(),
                                'Launch Count': int(transfer['Trip Count'].sum())
                            }
            
            if day_operations:
                # Create daily summary record
                day_record = {'Day': t}
                
                # Add information for each transport type
                for op_name, op_data in day_operations.items():
                    if 'Transport Time' in op_data:
                        day_record[op_name] = f"{op_data['Cargo']:.0f}t/{op_data['Launch Count']}trips({op_data['Transport Time']}d)"
                    else:
                        day_record[op_name] = f"{op_data['Cargo']:.0f}t/{op_data['Launch Count']}trips"
                
                detailed_schedule.append(day_record)
        
        if detailed_schedule:
            df = pd.DataFrame(detailed_schedule)
            
            # Calculate total statistics
            print("\nTransport Schedule Total Statistics:")
            print("-" * 50)
            
            # Summarize by transport type
            transport_summary = {}
            for record in detailed_schedule:
                for key, value in record.items():
                    if key != 'Day' and 't/' in str(value):
                        # Extract cargo and launch count
                        parts = value.split('t/')
                        cargo = float(parts[0])
                        trips = int(parts[1].split('trips')[0])
                        
                        transport_type = key
                        if transport_type not in transport_summary:
                            transport_summary[transport_type] = {'Total Cargo': 0, 'Total Launches': 0}
                        
                        transport_summary[transport_type]['Total Cargo'] += cargo
                        transport_summary[transport_type]['Total Launches'] += trips
            
            # Print summary information
            summary_df = pd.DataFrame([
                {
                    'Transport Type': transport_type,
                    'Total Cargo(tons)': data['Total Cargo'],
                    'Total Launches': data['Total Launches'],
                    'Avg Load/Flight(tons)': data['Total Cargo'] / data['Total Launches'] if data['Total Launches'] > 0 else 0
                }
                for transport_type, data in transport_summary.items()
            ])
            
            print(summary_df.round(1).to_string(index=False))
            
            # Total rocket launch statistics (Earth only)
            earth_rockets = self.results['Rocket Usage']['Total Rockets']
            print(f"\nEarth Rocket Launch Statistics:")
            print(f"  Total Launches: {earth_rockets:.0f}")
            print(f"  Avg Daily Launches: {self.results['Rocket Usage']['Average Daily Rockets']:.2f}")
            print(f"  Usage Rate: {self.results['Rocket Usage']['Rocket Usage Rate']:.2f}%")
            
            # Display detailed schedule (first 20 days and last 20 days)
            print("\nDetailed Launch Schedule (first 20 days and last 20 days):")
            print("-" * 80)
            
            if len(df) > 40:
                print("First 20 days:")
                print(df.head(20).to_string(index=False))
                print("\n...\n")
                print("Last 20 days:")
                print(df.tail(20).to_string(index=False))
            else:
                print(df.to_string(index=False))
            
            # Save as CSV file
            csv_filename = 'detailed_transport_schedule.csv'
            df.to_csv(csv_filename, index=False, encoding='utf-8')
            print(f"\nDetailed schedule saved as: {csv_filename}")
            
            return df, summary_df
        else:
            print("No transport records")
            return pd.DataFrame(), pd.DataFrame()
    
    def generate_all_visualizations(self):
        """Improved: Generate all visualization charts and tables with better organization"""
        print("\n" + "="*80)
        print("Generating Improved Visualization Charts and Tables")
        print("="*80)
        
        results = {}
        
        # 1. Plot combined inventory level changes line chart
        print("\n1. Generating combined inventory chart...")
        inventory_data = self.plot_inventory_levels()
        results['inventory_data'] = inventory_data
        
        # 2. Plot improved station operations charts
        print("\n2. Generating improved station operations charts...")
        station_ops_data = self.plot_station_operations()
        results['station_ops_data'] = station_ops_data
        
        # 3. Plot improved slow elevator launch schedule
        print("\n3. Generating improved slow elevator schedule...")
        slow_elevator_stats = self.plot_slow_elevator_schedule()
        results['slow_elevator_stats'] = slow_elevator_stats
        
        # 4. Generate Earth direct launch schedule timetable
        print("\n4. Generating Earth direct launch schedule...")
        earth_direct_schedule = self.generate_earth_direct_schedule_table()
        results['earth_direct_schedule'] = earth_direct_schedule
        
        # 5. Generate detailed launch schedule timetable
        print("\n5. Generating detailed launch schedule...")
        detailed_schedule, schedule_summary = self.generate_detailed_schedule_table()
        results['detailed_schedule'] = detailed_schedule
        results['schedule_summary'] = schedule_summary
        
        # 6. Plot clean launch schedule heatmap
        print("\n6. Generating clean heatmap...")
        heatmap_data = self.plot_launch_schedule_heatmap()
        results['heatmap_data'] = heatmap_data
        
        # 7. Generate summary report
        self.generate_summary_report()
        
        print("\n" + "="*80)
        print("All improved visualizations completed!")
        print("="*80)
        
        return results
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "="*80)
        print("Summary Report")
        print("="*80)
        
        costs = self.results['Costs']
        rocket_stats = self.results['Rocket Usage']
        inv_stats = self.results['Inventory Statistics']
        additional_costs = self.results['Additional Cost Analysis']
        
        print("\n1. Cost Analysis:")
        print(f"   Total Cost: ${costs['Total Cost (USD)']/1e9:.2f}B")
        print(f"   Fixed Transport: ${costs['Fixed Trip Cost (USD)']/1e9:.2f}B")
        print(f"   Variable Transport: ${costs['Variable Transport Cost (USD)']/1e9:.2f}B")
        print(f"   Inventory Penalty: ${costs['Inventory Deviation Penalty (USD)']/1e9:.2f}B")
        print(f"   Daily Avg Cost: ${costs['Daily Average Cost (USD)']/1e6:.2f}M")
        
        print("\n2. Inventory Statistics:")
        print(f"   Moon Inventory Mean: {inv_stats['Moon Inventory Mean']:.0f} t")
        print(f"   Moon Inventory Std: {inv_stats['Moon Inventory Std']:.0f} t")
        print(f"   Moon Inventory Range: [{inv_stats['Moon Inventory Min']:.0f}, {inv_stats['Moon Inventory Max']:.0f}] t")
        
        print("\n3. Rocket Usage:")
        print(f"   Total Launches: {rocket_stats['Total Rockets']:.0f}")
        print(f"   Avg Daily: {rocket_stats['Average Daily Rockets']:.2f}")
        print(f"   Usage Rate: {rocket_stats['Rocket Usage Rate']:.1f}%")
        
        print("\n4. Efficiency Metrics:")
        total_transport = additional_costs['Transport Efficiency Metrics']['Total Transport (tons)']
        cost_per_ton = additional_costs['Cost per Ton (USD/ton)']
        print(f"   Total Transport: {total_transport:.0f} t")
        print(f"   Cost per Ton: ${cost_per_ton/1e3:.1f}K/t")
        print(f"   Demand Satisfaction: {total_transport/(self.daily_consumption * self.T)*100:.1f}%")
        
        print("\n5. Generated Files:")
        print("   - inventory_levels_combined.png (Combined inventory chart)")
        print("   - station_operations_improved.png (Station operations)")
        print("   - slow_elevator_schedule_improved.png (Slow elevator schedule)")
        print("   - launch_schedule_heatmap_clean.png (Launch heatmap)")
        print("   - earth_direct_schedule.csv (Earth direct schedule)")
        print("   - detailed_transport_schedule.csv (Detailed schedule)")
        
        print("="*80)


def main():
    """Main function: Run model and generate all visualizations"""
    print("Earth-Moon Logistics Scheduling Model - 54-day Steady-state Cycle Model")
    print("="*80)
    print("Model Parameters:")
    print(f"  Cycle Days: 54 days")
    print(f"  Moon Daily Consumption: 1,500 tons")
    print(f"  Earth Rocket Limit: 10 rockets/day")
    print(f"  Slow Elevator Limit: 1 per station/day, 6-day interval")
    print(f"  Initial Inventory: 0 tons")
    print(f"  Inventory Penalty Weight λ: 1e6")
    print("="*80)
    
    # Create 54-day cycle model
    model = LunarLogistics54DayModel(
        lambda_weight=1e6,  # Inventory deviation penalty weight
        T=54                # 54-day cycle
    )
    
    # Build model
    print("\nBuilding model...")
    model.build_model()
    
    # Solve model
    print("\nSolving model...")
    if model.solve(time_limit=18000, verbose=True):  # 修改为5小时（18000秒）
        # Generate all visualization charts and tables
        print("\nGenerating improved visualizations...")
        visualization_results = model.generate_all_visualizations()
        
        return model, visualization_results
    else:
        print("Model solving failed")
        return None, None


if __name__ == "__main__":
    # Run model and generate visualizations
    solution, viz_results = main()
    
    if solution:
        print("\n" + "="*80)
        print("Program execution completed successfully!")
        print("="*80)
        print("\nSummary of generated files:")
        print("1. inventory_levels_combined.png - Combined inventory chart")
        print("2. station_operations_improved.png - Station operations analysis")
        print("3. slow_elevator_schedule_improved.png - Slow elevator schedule")
        print("4. launch_schedule_heatmap_clean.png - Launch schedule heatmap")
        print("5. earth_direct_schedule.csv - Earth direct launch timetable")
        print("6. detailed_transport_schedule.csv - Detailed transport schedule")
        print("\nAll files have been saved with improved visual styling.")
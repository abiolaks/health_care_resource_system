class ResourceOptimizer:
    """Optimization engine with guaranteed feasibility"""
    def __init__(self, data):
        self.data = data
        self.problem = pulp.LpProblem("Healthcare_Resource_Allocation", pulp.LpMinimize)
        self.solution = None
        self.min_budget = 0.0
        self.emergency_mode = False
        
        # Initialize costs and requirements
        self.costs = {
            'Beds': 100000,    # Ultra basic beds
            'Doctors': 500000, # Community health workers
            'Vaccines': 200    # Subsidized vaccines
        }
        self.min_requirements = {
            'Beds': 0.0002,    # 1 bed per 5000 people
            'Doctors': 0.0001, # 1 doctor per 10,000 people
            'Vaccines': 0.05   # Vaccinate 5% population
        }

    def calculate_min_budget(self):
        """Calculate minimum required budget"""
        total_pop = self.data['Population Size'].sum()
        self.min_budget = sum(
            total_pop * req * self.costs[res]
            for res, req in self.min_requirements.items()
        )
        return self.min_budget

    def build_model(self, total_budget):
        # Ensure minimum budget is at least 1 NGN to prevent division by zero
        self.min_budget = max(self.min_budget, 1.0)
        
        # Calculate budget ratio safely
        budget_ratio = min(total_budget / self.min_budget, 1.0) if self.min_budget > 0 else 0.0
        
        regions = self.data['Region Name/Code'].unique()
        resources = ['Beds', 'Doctors', 'Vaccines']
        
        # Create decision variables (using Integer type if necessary)
        self.vars = pulp.LpVariable.dicts(
            "Allocation",
            [(r, res) for r in regions for res in resources],
            lowBound=0,
            cat='Integer'
        )
        
        # Objective: Prioritize by crisis index
        self.problem += pulp.lpSum(
            (self.data.loc[self.data['Region Name/Code'] == r, 'Crisis_Index'].values[0]/100) *
            (self.vars[(r, 'Beds')] + self.vars[(r, 'Doctors')] + self.vars[(r, 'Vaccines')])
            for r in regions
        )
        
        # Budget constraint
        self.problem += pulp.lpSum(
            self.costs[res] * self.vars[(r, res)]
            for r in regions for res in resources
        ) <= total_budget
        
        # Dynamic constraints
        self.emergency_mode = total_budget < self.min_budget
        budget_ratio = min(total_budget / self.min_budget, 1.0)

        for r in regions:
            pop = self.data.loc[self.data['Region Name/Code'] == r, 'Population Size'].values[0]
            for res in resources:
                min_req = pop * self.min_requirements[res] * budget_ratio
                self.problem += self.vars[(r, res)] >= max(min_req, 1)

    def solve(self):
        """Solve the optimization problem"""
        self.problem.solve()
        if pulp.LpStatus[self.problem.status] in ['Optimal', 'Feasible']:
            # Use the original keys (tuples) from self.vars
            self.solution = {
                key: var.varValue for key, var in self.vars.items() if var.varValue > 0
            }
            return True
        return False

    def get_explanations(self):
        """Generate allocation explanations"""
        if not self.solution:
            return "No solution found"
            
        allocations = pd.DataFrame.from_dict(
            {(k[0], k[1]): v for k, v in self.solution.items()},
            orient='index',
            columns=['Quantity']
        ).reset_index()
        
        allocations.columns = ['Region', 'Resource', 'Quantity']
        
        # Merge with original data
        result = pd.merge(
            allocations.pivot(index='Region', columns='Resource', values='Quantity'),
            self.data[['Region Name/Code', 'Crisis_Index', 'Population Size']],
            left_on='Region',
            right_on='Region Name/Code'
        )
        
        # Calculate coverage metrics
        result['Beds_per_5000'] = result['Beds'] / (result['Population Size'] / 5000)
        result['Doctors_per_10k'] = result['Doctors'] / (result['Population Size'] / 10000)
        result['Vaccination_Coverage'] = result['Vaccines'] / result['Population Size']
        
        return result.sort_values('Crisis_Index', ascending=False)

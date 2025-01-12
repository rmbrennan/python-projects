class TransferAgent(FPLAgent):
    def __init__(self):
        config = AgentConfig(
            name="TransferAgent",
            provides={"transfer_suggestions"},
            requires={"player_data", "optimal_team"}  # Requires both previous agents
        )
        super().__init__(config)
        
    def validate(self) -> bool:
        if not self.validate_dependencies():
            return False
        return True
        
    def process(self) -> pd.DataFrame:
        # Get data from dependencies
        scraper = self.get_dependency("DataScraper")
        optimizer = self.get_dependency("TeamOptimizer")
        
        if not scraper or not optimizer:
            raise ValueError("Required dependencies not found")
            
        data = scraper.process()
        optimal_team = optimizer.process()
        
        return self._suggest_transfers(data, optimal_team)
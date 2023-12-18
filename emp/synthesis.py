from abc import ABC, abstractmethod

# Tool interfaces
class SYBATool(ABC):
    @abstractmethod
    def predict_syba_score(self, molecule):
        pass

class SCScoreTool(ABC):
    @abstractmethod
    def predict_scscore(self, molecule):
        pass

class SAScoreTool(ABC):
    @abstractmethod
    def predict_sascore(self, molecule):
        pass

# Concrete implementations of the tools
class SYBA(SYBATool):
    def predict_syba_score(self, molecule):
        # Implementation of SYBA score prediction
        return 0.75

class SCScore(SCScoreTool):
    def predict_scscore(self, molecule):
        # Implementation of SCScore prediction
        return 0.80

class SAScore(SAScoreTool):
    def predict_sascore(self, molecule):
        # Implementation of SAScore prediction
        return 0.85

# Synthesizability Score Adapter
class SynthesizabilityScoreAdapter:
    def __init__(self, tool, adapter_method):
        self.tool = tool
        self.adapter_method = adapter_method

    def predict_synthesizability_score(self, molecule):
        if self.adapter_method == 'syba':
            return self.tool.predict_syba_score(molecule)
        elif self.adapter_method == 'scscore':
            return self.tool.predict_scscore(molecule)
        elif self.adapter_method == 'sascore':
            return self.tool.predict_sascore(molecule)
        else:
            raise ValueError(f"Invalid adapter method: {self.adapter_method}")

# Tool Factory
class ToolFactory:
    def create_tool(self, tool_type):
        if tool_type == 'syba':
            return SynthesizabilityScoreAdapter(SYBA(), 'syba')
        elif tool_type == 'scscore':
            return SynthesizabilityScoreAdapter(SCScore(), 'scscore')
        elif tool_type == 'sascore':
            return SynthesizabilityScoreAdapter(SAScore(), 'sascore')
        else:
            raise ValueError(f"Invalid tool type: {tool_type}")

# Example usage
if __name__ == "__main__":
    tool_factory = ToolFactory()

    # Use SYBA for synthesizability prediction
    syba_tool = tool_factory.create_tool('syba')
    syba_score = syba_tool.predict_synthesizability_score("molecule1")
    print(f"SYBA Synthesizability Score: {syba_score}")

    # Use SCScore for synthesizability prediction
    scscore_tool = tool_factory.create_tool('scscore')
    scscore = scscore_tool.predict_synthesizability_score("molecule2")
    print(f"SCScore Synthesizability Score: {scscore}")

    # Use SAScore for synthesizability prediction
    sascore_tool = tool_factory.create_tool('sascore')
    sascore = sascore_tool.predict_synthesizability_score("molecule3")
    print(f"SAScore Synthesizability Score: {sascore}")

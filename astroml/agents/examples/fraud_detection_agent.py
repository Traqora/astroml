"""Example fraud detection agent using the Agent framework."""

from ..agent import Agent, AgentConfig
from ..memory import ConversationMemory
from ..planning import HeuristicPlanner
from ..tools import DataAnalyzer, DataLoader, GraphBuilder, GraphAnalyzer


def create_fraud_detection_agent() -> Agent:
    """Create a fraud detection agent with relevant tools.

    Returns:
        Configured Agent instance
    """
    config = AgentConfig(
        name="fraud-detection-agent",
        model="gpt-4",
        temperature=0.3,  # Lower temperature for more deterministic fraud detection
        max_steps=15,
        enable_planning=True,
        enable_memory=True,
        verbose=True,
    )

    # Initialize components
    memory = ConversationMemory(max_history=50)
    planner = HeuristicPlanner()

    # Create tools for fraud detection
    tools = [
        DataLoader(),
        DataAnalyzer(),
        GraphBuilder(),
        GraphAnalyzer(),
    ]

    return Agent(
        config=config,
        tools=tools,
        memory=memory,
        planner=planner,
    )


def main():
    """Run example fraud detection task."""
    agent = create_fraud_detection_agent()

    task = """
    Detect potential fraud patterns in the Stellar transaction data.
    Steps:
    1. Load the transaction data from the database
    2. Build a transaction graph with accounts as nodes
    3. Analyze the graph for suspicious patterns
    4. Identify high-risk accounts based on graph metrics
    5. Generate a fraud detection report
    """

    context = {
        "data_source": "postgresql://localhost/astroml",
        "time_window": "30d",
        "risk_threshold": 0.8,
    }

    result = agent.run(task, context)

    print("\n" + "=" * 50)
    print("FRAUD DETECTION RESULTS")
    print("=" * 50)
    print(f"Success: {result.success}")
    print(f"Final Answer: {result.final_answer}")
    print(f"\nExecution Steps: {len(result.steps)}")

    for step in result.steps:
        print(f"\nStep {step.step_number}:")
        print(f"  Thought: {step.thought[:100]}...")
        if step.action:
            print(f"  Action: {step.action}")
        if step.observation:
            print(f"  Observation: {step.observation[:100]}...")

    if result.error:
        print(f"\nError: {result.error}")


if __name__ == "__main__":
    main()

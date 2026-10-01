"""Example model training agent using the Agent framework."""

from ..agent import Agent, AgentConfig
from ..memory import ConversationMemory
from ..planning import HierarchicalPlanner
from ..tools import DataLoader, DataAnalyzer, ModelTrainer, ModelEvaluator


def create_model_training_agent() -> Agent:
    """Create a model training agent with relevant tools.

    Returns:
        Configured Agent instance
    """
    config = AgentConfig(
        name="model-training-agent",
        model="gpt-4",
        temperature=0.5,
        max_steps=20,
        enable_planning=True,
        enable_memory=True,
        verbose=True,
    )

    # Initialize components
    memory = ConversationMemory(max_history=100)
    planner = HierarchicalPlanner(max_depth=2)

    # Create tools for model training
    tools = [
        DataLoader(),
        DataAnalyzer(),
        ModelTrainer(),
        ModelEvaluator(),
    ]

    return Agent(
        config=config,
        tools=tools,
        memory=memory,
        planner=planner,
    )


def main():
    """Run example model training task."""
    agent = create_model_training_agent()

    task = """
    Train a Graph Neural Network model for fraud detection on the Stellar network.
    The model should:
    1. Load the transaction graph data
    2. Prepare features for node classification
    3. Train a GraphSAGE model
    4. Evaluate the model on test data
    5. Save the trained model for deployment
    """

    context = {
        "graph_data": "data/transaction_graph.parquet",
        "model_type": "sage",
        "hidden_dim": 128,
        "num_layers": 3,
        "learning_rate": 0.001,
        "epochs": 100,
    }

    result = agent.run(task, context)

    print("\n" + "=" * 50)
    print("MODEL TRAINING RESULTS")
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

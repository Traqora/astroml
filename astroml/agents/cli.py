"""CLI interface for interacting with agents."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from .agent import Agent, AgentConfig
from .examples.fraud_detection_agent import create_fraud_detection_agent
from .examples.model_training_agent import create_model_training_agent
from .memory import ConversationMemory, VectorMemory
from .planning import HeuristicPlanner, HierarchicalPlanner
from .tools import (
    DataAnalyzer,
    DataLoader,
    DataWriter,
    GraphAnalyzer,
    GraphBuilder,
    ModelEvaluator,
    ModelTrainer,
)


def create_agent_from_type(agent_type: str) -> Agent:
    """Create an agent from a predefined type.

    Args:
        agent_type: Type of agent to create

    Returns:
        Configured Agent instance
    """
    agent_map = {
        "fraud-detection": create_fraud_detection_agent,
        "model-training": create_model_training_agent,
    }

    if agent_type in agent_map:
        return agent_map[agent_type]()

    # Default generic agent
    config = AgentConfig(name="generic-agent", verbose=True)
    tools = [
        DataLoader(),
        DataAnalyzer(),
        DataWriter(),
        GraphBuilder(),
        GraphAnalyzer(),
        ModelTrainer(),
        ModelEvaluator(),
    ]
    return Agent(
        config=config,
        tools=tools,
        memory=ConversationMemory(),
        planner=HeuristicPlanner(),
    )


def run_interactive(agent: Agent) -> None:
    """Run agent in interactive mode.

    Args:
        agent: Agent instance
    """
    print(f"Starting interactive session with {agent.config.name}")
    print("Type 'exit' or 'quit' to end the session\n")

    while True:
        try:
            task = input("You: ").strip()

            if task.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            if not task:
                continue

            print(f"\n{agent.config.name} is thinking...")
            result = agent.run(task)

            print(f"\nAgent: {result.final_answer}")

            if result.error:
                print(f"Error: {result.error}")

            if agent.config.verbose and len(result.steps) > 1:
                print(f"\n(Executed {len(result.steps)} steps)")

            print()

        except KeyboardInterrupt:
            print("\nInterrupted. Type 'exit' to quit.")
        except EOFError:
            print("\nGoodbye!")
            break


def run_single_task(agent: Agent, task: str, context: Optional[str] = None) -> None:
    """Run agent on a single task.

    Args:
        agent: Agent instance
        task: Task description
        context: Optional context as JSON string
    """
    context_dict = json.loads(context) if context else None

    print(f"Running task with {agent.config.name}...")
    result = agent.run(task, context_dict)

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Final Answer:\n{result.final_answer}")

    if result.error:
        print(f"\nError: {result.error}")

    print(f"\nExecution Steps: {len(result.steps)}")

    if agent.config.verbose:
        for step in result.steps:
            print(f"\nStep {step.step_number}:")
            print(f"  Thought: {step.thought}")
            if step.action:
                print(f"  Action: {step.action}")
            if step.tool_call:
                print(f"  Tool Call: {json.dumps(step.tool_call, indent=2)}")
            if step.observation:
                print(f"  Observation: {step.observation}")

    # Print execution trace
    print("\n" + "=" * 60)
    print("EXECUTION TRACE")
    print("=" * 60)
    trace = agent.get_execution_trace()
    print(json.dumps(trace, indent=2))


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point.

    Args:
        argv: Command line arguments

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="astroml-agent",
        description="AstroML Agent Framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive", help="Run agent in interactive mode"
    )
    interactive_parser.add_argument(
        "--agent-type",
        choices=["fraud-detection", "model-training", "generic"],
        default="generic",
        help="Type of agent to use",
    )

    # Run command
    run_parser = subparsers.add_parser("run", help="Run agent on a single task")
    run_parser.add_argument(
        "--agent-type",
        choices=["fraud-detection", "model-training", "generic"],
        default="generic",
        help="Type of agent to use",
    )
    run_parser.add_argument("--task", required=True, help="Task description")
    run_parser.add_argument("--context", help="Context as JSON string")
    run_parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    # List tools command
    subparsers.add_parser("list-tools", help="List available tools")

    args = parser.parse_args(argv)

    if args.command == "interactive":
        agent = create_agent_from_type(args.agent_type)
        run_interactive(agent)
        return 0

    if args.command == "run":
        agent = create_agent_from_type(args.agent_type)
        agent.config.verbose = args.verbose
        run_single_task(agent, args.task, args.context)
        return 0

    if args.command == "list-tools":
        agent = create_agent_from_type("generic")
        tools = agent.tool_registry.get_all_tools()
        print("Available Tools:")
        print("=" * 60)
        for tool in tools:
            print(f"\n{tool.name}")
            print(f"  Description: {tool.description}")
            if tool.parameters:
                print(f"  Parameters:")
                for param_name, param_schema in tool.parameters.items():
                    required = " (required)" if param_schema.get("required") else ""
                    print(f"    - {param_name}: {param_schema.get('type', 'any')}{required}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

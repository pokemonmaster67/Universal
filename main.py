import google.generativeai as genai
import os
from typing import List, Dict, Tuple
import textwrap
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import time
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from collections import deque

class ThoughtVisualizer:
    def __init__(self):
        self.thought_graph = nx.DiGraph()
        self.thought_queue = deque(maxlen=10)

    def add_thought(self, thought: str, category: str):
        """Add a thought node to the graph."""
        timestamp = time.time()
        self.thought_graph.add_node(timestamp, thought=thought, category=category)
        if len(self.thought_queue) > 0:
            self.thought_graph.add_edge(list(self.thought_queue)[-1], timestamp)
        self.thought_queue.append(timestamp)

    def generate_visualization(self) -> str:
        """Generate an ASCII visualization of the thought process."""
        if len(self.thought_graph) == 0:
            return "No thoughts processed yet."

        visualization = []
        for node in self.thought_queue:
            thought = self.thought_graph.nodes[node]['thought']
            category = self.thought_graph.nodes[node]['category']
            visualization.append(f"[{category}] {thought}")

        return "\n└─> ".join(visualization)

class ScienceAnalyzer:
    """Analyzes responses through scientific principles."""

    def __init__(self):
        self.principles = {
            'quantum': ['superposition', 'entanglement', 'uncertainty'],
            'thermo': ['entropy', 'energy conservation', 'heat transfer'],
            'relativity': ['time dilation', 'mass-energy', 'gravity'],
            'biology': ['evolution', 'adaptation', 'homeostasis']
        }

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text for scientific principle applications."""
        scores = {}
        for category, principles in self.principles.items():
            score = sum(principle.lower() in text.lower() for principle in principles)
            scores[category] = score / len(principles)
        return scores

class UniversalAgent:
    def __init__(self, api_key: str):
        """Initialize the Universal agent with enhanced capabilities."""
        self.console = Console()
        self.layout = Layout()
        genai.configure(api_key=api_key)

        # Initialize components
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
        self.thought_visualizer = ThoughtVisualizer()
        self.science_analyzer = ScienceAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=3)

        # Load system prompt
        with open('universal_prompt.md', 'r') as f:
            self.system_prompt = f.read()

        # Initialize chat and response history
        self.chat = self.model.start_chat(history=[])
        self.response_history = []
        self._initialize_chat()

    def _initialize_chat(self):
        """Initialize the chat with enhanced monitoring."""
        try:
            self.console.print("[cyan]Initializing quantum consciousness...[/cyan]")
            self.chat.send_message(self.system_prompt)
            self.thought_visualizer.add_thought(
                "System initialization complete", "SYSTEM"
            )
        except Exception as e:
            self.console.print(f"[red]Initialization Error: {str(e)}[/red]")
            raise

    def _analyze_response(self, response: str) -> Dict:
        """Analyze response using scientific principles."""
        # Scientific principle analysis
        science_scores = self.science_analyzer.analyze_text(response)

        # Complexity analysis
        complexity_score = len(set(response.split())) / len(response.split())

        return {
            'science_scores': science_scores,
            'complexity': complexity_score
        }

    def _generate_thought_map(self) -> Panel:
        """Generate a visual map of the thought process."""
        thought_viz = self.thought_visualizer.generate_visualization()
        return Panel(
            Markdown(thought_viz),
            title="Thought Process Map",
            border_style="cyan"
        )

    def _create_science_table(self, scores: Dict[str, float]) -> Table:
        """Create a table showing scientific principle application."""
        table = Table(title="Scientific Principle Analysis")
        table.add_column("Principle", style="cyan")
        table.add_column("Application Score", style="magenta")

        for principle, score in scores.items():
            table.add_row(principle, f"{score:.2f}")

        return table

    def process_query(self, query: str) -> str:
        """Process a query with enhanced analysis and visualization."""
        try:
            # Add query to thought process
            self.thought_visualizer.add_thought(f"Query received: {query}", "INPUT")

            # Use a single progress display for the entire process
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True
            ) as progress:
                # Generate response
                task1 = progress.add_task("Generating response...", total=100)
                response = self.chat.send_message(query)
                progress.update(task1, completed=100)

                # Analyze response
                task2 = progress.add_task("Analyzing response...", total=100)
                analysis = self._analyze_response(response.text)
                progress.update(task2, completed=100)

            # Create and display results
            self.console.print("\n[bold cyan]Response:[/bold cyan]")
            self.console.print(Panel(Markdown(response.text)))

            self.console.print("\n[bold cyan]Thought Process:[/bold cyan]")
            self.console.print(self._generate_thought_map())

            self.console.print("\n[bold cyan]Analysis:[/bold cyan]")
            self.console.print(self._create_science_table(analysis['science_scores']))

            # Add to thought process
            self.thought_visualizer.add_thought(
                f"Response generated (complexity: {analysis['complexity']:.2f})",
                "OUTPUT"
            )

            return response.text

        except Exception as e:
            error_msg = f"Error in processing: {str(e)}"
            self.console.print(f"[red]{error_msg}[/red]")
            return error_msg

    def run_interactive(self):
        """Run the agent with enhanced interactive features."""
        welcome_panel = Panel(
            "[bold blue]Universal Agent - Advanced World of Thought System[/bold blue]\n"
            "Quantum-enabled reasoning engine with advanced scientific analysis.\n"
            "Type 'exit' to quit, 'stats' for session statistics, 'viz' for thought visualization.",
            border_style="blue"
        )
        self.console.print(welcome_panel)

        while True:
            try:
                query = Prompt.ask("\n[bold green]Enter your query")

                if query.lower() == 'exit':
                    self.console.print("[yellow]Shutting down quantum systems... Goodbye![/yellow]")
                    break
                elif query.lower() == 'stats':
                    self._display_session_stats()
                    continue
                elif query.lower() == 'viz':
                    self.console.print(self._generate_thought_map())
                    continue

                self.process_query(query)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Emergency shutdown initiated... Goodbye![/yellow]")
                break
            except Exception as e:
                self.console.print(f"[red]Critical error: {str(e)}[/red]")

    def _display_session_stats(self):
        """Display session statistics."""
        stats_table = Table(title="Session Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="magenta")

        num_queries = len(self.thought_visualizer.thought_queue)
        avg_response_time = np.mean([
            node[1]['response_time']
            for node in self.thought_visualizer.thought_graph.nodes(data=True)
            if 'response_time' in node[1]
        ]) if num_queries > 0 else 0

        stats_table.add_row("Total Queries", str(num_queries))
        stats_table.add_row("Average Response Time", f"{avg_response_time:.2f}s")

        self.console.print(stats_table)

def main():
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Please set the GOOGLE_API_KEY environment variable")
        return

    try:
        agent = UniversalAgent(api_key)
        agent.run_interactive()
    except Exception as e:
        print(f"Critical system error: {str(e)}")

if __name__ == "__main__":
    main()
"""CLI entry point, interactive event loop, and multi-core parallel training for Google Dino AI."""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List
import multiprocessing
import pygame

from config.settings import GameConfig
from game.engine import GameEngine
from rendering.renderer import Renderer
from core.genome import Genome
from telemetry.plotter import TelemetryPlotter


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Google Dinosaur AI - Advanced Python Edition"
    )
    parser.add_argument(
        "--mode",
        choices=["training", "play", "evaluation", "headless", "versus"],
        default="training",
        help="Game mode: 'training' (2000 dinos), 'play' (human), 'evaluation' (watch AI), 'headless' (turbo), 'versus' (human vs AI).",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Population size (default: 2000 for training, 2 for versus, 1 for play/eval).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to pre-trained model file (legacy 'rede' binary or JSON).",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=0,
        help="Target number of generations for headless mode (0 = infinite).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of CPU cores for parallel headless training (default: 1).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target rendering FPS (default: 60).",
    )
    parser.add_argument(
        "--mute",
        action="store_true",
        help="Start with audio muted.",
    )
    parser.add_argument(
        "--export-plot",
        action="store_true",
        help="Generate and save an analytical evolution plot with Matplotlib after headless training.",
    )
    parser.add_argument(
        "--seed-best",
        action="store_true",
        help="Inicia a população semeada a partir do melhor dinossauro campeão histórico em vez de pesos aleatórios.",
    )
    return parser.parse_args()


def load_model_if_provided(engine: GameEngine, model_path_str: Optional[str], seed_best: bool = False) -> None:
    """Loads weights from file into engine."""
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    models_dir = base_dir / "models"
    best_file = models_dir / "champion_best.json"

    if not model_path_str:
        if engine.mode in ("play", "evaluation", "versus") or seed_best:
            if best_file.exists():
                model_path_str = str(best_file)
            else:
                json_models = sorted(models_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                if json_models:
                    model_path_str = str(json_models[0])
                elif (models_dir / "rede_legacy_x1").exists():
                    model_path_str = str(models_dir / "rede_legacy_x1")
                else:
                    return
        else:
            return

    path = Path(model_path_str)
    if not path.exists():
        fallback_path = base_dir / model_path_str
        if fallback_path.exists():
            path = fallback_path
        else:
            print(f"[Aviso] Arquivo de modelo não encontrado: {path}")
            return

    try:
        if path.suffix == ".json":
            genome = Genome.from_json(path)
        else:
            genome = Genome.from_legacy_binary(path)

        print(f"[Sucesso] Modelo Campeão carregado de '{path.name}' ({genome.length} pesos)!")

        if engine.single_brain:
            engine.single_brain.load_genome(genome)
        if engine.ga:
            engine.ga.seed_all_from_best(genome)
            if engine.batch_brain:
                engine.batch_brain.load_population_genomes(engine.ga.population)

    except Exception as e:
        print(f"[Erro] Falha ao carregar modelo '{path}': {e}")


def _parallel_worker(core_id: int, population_size: int, target_generations: int) -> float:
    """Headless worker process running simulation on an isolated CPU core."""
    config = GameConfig()
    config.genetic.population_size = population_size
    engine = GameEngine(config=config, mode="training", enable_telemetry=(core_id == 0))

    gen = 0
    while gen < target_generations:
        finished = engine.step(dt=0.005)
        if finished:
            gen += 1
            if core_id == 0:
                print(f"[Core {core_id}] Geração {gen}/{target_generations} - Recorde: {engine.record_distance:.0f}px")

    if core_id == 0 and engine.ga and len(engine.ga.population) > 0:
        champion = engine.ga.population[0]
        out_path = Path(f"models/parallel_champion_record_{engine.record_distance:.0f}.json")
        champion.to_json(out_path, metadata={"record": engine.record_distance, "generations": gen})

    return engine.record_distance


def run_parallel_training(num_cores: int, population: int, generations: int, export_plot: bool = False) -> None:
    """Orchestrates multi-core parallel simulations across available CPU cores."""
    gens = generations if generations > 0 else 10
    print(f"\nIniciando Treinamento Paralelo Multi-Core em {num_cores} núcleos...")
    print(f"População por núcleo: {population} | Meta: {gens} gerações")
    t0 = time.perf_counter()

    with multiprocessing.Pool(processes=num_cores) as pool:
        args = [(i, population, gens) for i in range(num_cores)]
        results = pool.starmap(_parallel_worker, args)

    total_time = time.perf_counter() - t0
    best_overall = max(results)
    print(f"\nTreinamento Paralelo Concluído em {total_time:.2f}s!")
    print(f"Melhor distância alcançada entre os núcleos: {best_overall:.1f} pixels")

    if export_plot:
        plotter = TelemetryPlotter()
        plotter.generate_report()


def run_headless(engine: GameEngine, max_generations: int = 10, export_plot: bool = False) -> None:
    """Runs high-performance single-thread simulation without graphical overhead."""
    print(f"\nIniciando treinamento Headless (População: {engine.population_size})...")
    gen = 0
    t_start = time.perf_counter()

    try:
        while True:
            finished = engine.step(dt=0.005)
            if finished:
                gen += 1
                best_fit = engine.ga.best_fitness_history[-1] if engine.ga else 0.0
                avg_fit = engine.ga.avg_fitness_history[-1] if engine.ga else 0.0
                elapsed = time.perf_counter() - t_start

                print(
                    f"Geração {gen:3d} | Recorde: {engine.record_distance:8.1f} px | "
                    f"Melhor Fit: {best_fit:8.1f} | Média Fit: {avg_fit:8.1f} | Tempo: {elapsed:.2f}s"
                )

                if max_generations > 0 and gen >= max_generations:
                    print(f"\nMeta de {max_generations} gerações atingida com sucesso!")
                    break

    except KeyboardInterrupt:
        print("\nTreinamento interrompido pelo usuário.")

    if engine.ga and len(engine.ga.population) > 0:
        champion = engine.ga.population[0]
        out_path = Path(f"models/champion_gen{gen}_{engine.record_distance:.0f}.json")
        champion.to_json(out_path, metadata={"generation": gen, "record_distance": engine.record_distance})
        print(f"Melhor modelo salvo em: {out_path}")

    if export_plot:
        plotter = TelemetryPlotter()
        plotter.generate_report()


def run_graphical(engine: GameEngine, target_fps: int = 60, start_muted: bool = False) -> None:
    """Runs interactive graphical simulation with Pygame."""
    renderer = Renderer(config=engine.config.display)
    renderer.initialize_window()
    if start_muted:
        renderer.audio.muted = True

    clock = pygame.time.Clock()
    clock_period = 0.005

    running = True
    player_jump = False
    player_duck = False
    player_airplane = False

    sim_speed = 1
    speed_levels = [1, 2, 5, 10, 20, 50]

    print("\nControles e Atalhos:")
    print("  ESC: Alternar desenho da tela (Modo Turbo Ultra Rápido)")
    print("  + / - ou Seta Cima/Baixo: Acelerar simulação (1x, 2x, 5x, 10x, 20x, 50x)")
    print("  S: Alternar Raios de Sensores da IA (On/Off)")
    print("  H: Alternar Caixas de Colisão / Hitbox (On/Off)")
    print("  N: Alternar Ciclo Dia/Noite (Modo Escuro On/Off)")
    print("  M: Alternar Áudio / Mudo")
    if engine.mode in ("play", "versus"):
        print("  Seta Cima: Pular | Seta Baixo: Abaixar | Barra de Espaço: Avião")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    renderer.show_display = not renderer.show_display
                    status = "Visível" if renderer.show_display else "Oculto (Turbo Ultra Rápido ~50x)"
                    print(f"[Visualização] {status}")

                elif event.key == pygame.K_s:
                    renderer.show_sensors = not renderer.show_sensors
                    print(f"[Raios de Sensores] {'Ativados' if renderer.show_sensors else 'Desativados'}")

                elif event.key == pygame.K_h:
                    renderer.show_hitboxes = not renderer.show_hitboxes
                    print(f"[Hitboxes] {'Visíveis' if renderer.show_hitboxes else 'Ocultas'}")

                elif event.key == pygame.K_m:
                    muted = renderer.audio.toggle_mute()
                    print(f"[Áudio] {'Mudo' if muted else 'Ativado'}")

                elif event.key == pygame.K_n:
                    renderer.enable_day_night = not renderer.enable_day_night
                    print(f"[Ciclo Dia/Noite] {'Ativado' if renderer.enable_day_night else 'Desativado (Modo Claro Padrão)'}")

                if engine.mode == "training":
                    if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS, pygame.K_UP):
                        idx = next((i for i, s in enumerate(speed_levels) if s > sim_speed), len(speed_levels) - 1)
                        sim_speed = speed_levels[idx]
                        print(f"[Simulação] Acelerada para {sim_speed}x")
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
                        idx = next((i for i, s in reversed(list(enumerate(speed_levels))) if s < sim_speed), 0)
                        sim_speed = speed_levels[idx]
                        print(f"[Simulação] Reduzida para {sim_speed}x")
                elif engine.mode in ("play", "versus"):
                    if event.key == pygame.K_UP:
                        player_jump = True
                    elif event.key == pygame.K_DOWN:
                        player_duck = True
                    elif event.key == pygame.K_SPACE:
                        player_airplane = True

            elif event.type == pygame.KEYUP:
                if engine.mode in ("play", "versus"):
                    if event.key == pygame.K_UP:
                        player_jump = False
                    elif event.key == pygame.K_DOWN:
                        player_duck = False
                    elif event.key == pygame.K_SPACE:
                        player_airplane = False

        # Execute physics ticks based on speed multiplier or Turbo mode
        current_steps = 50 if not renderer.show_display else sim_speed
        p_inputs = (player_jump, player_duck, player_airplane) if engine.mode in ("play", "versus") else None

        for _ in range(current_steps):
            finished = engine.step(dt=clock_period, player_inputs=p_inputs)
            if finished:
                break

        # Render frame (if display is active)
        if renderer.show_display:
            renderer.render(engine, clock_period=clock_period, sim_speed=sim_speed)
            clock.tick(target_fps)
        else:
            # In Turbo mode, avoid 60 FPS lock and tick minimally for OS window event handling
            clock.tick(500)

    pygame.quit()


def main() -> None:
    args = parse_arguments()

    if args.parallel > 1 and args.mode == "headless":
        pop = args.population or 500
        run_parallel_training(
            num_cores=args.parallel,
            population=pop,
            generations=args.generations,
            export_plot=args.export_plot,
        )
        return

    config = GameConfig()
    if args.population is not None:
        config.genetic.population_size = args.population

    mode_map = {
        "training": "training",
        "play": "playable",
        "evaluation": "evaluation",
        "headless": "training",
        "versus": "versus",
    }
    engine_mode = mode_map[args.mode]
    engine = GameEngine(config=config, mode=engine_mode, enable_telemetry=True)

    load_model_if_provided(engine, args.model, seed_best=args.seed_best)

    if args.mode == "headless":
        run_headless(engine, max_generations=args.generations, export_plot=args.export_plot)
    else:
        run_graphical(engine, target_fps=args.fps, start_muted=args.mute)


if __name__ == "__main__":
    main()

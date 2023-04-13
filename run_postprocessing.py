
import os
import sys
import click
import subprocess
from loguru import logger


# get the path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# append the current directory to the sys.path list
sys.path.append(current_dir)

# NOTE: needed as we import this file from various locations
from model.config import OUTPUT_DIR  # noqa: E402
import model.output as output  # noqa: E402


def do_all(path: str, **kwargs) -> None:
    """
    First reads the input file for the client and generates extensions
    and cleans up all of the teeth. Also generates vectors for the teeth
    seating.

    Then runs another script to combine all generated stl files into a single
    minimesh and outputs it into the same folder
    """
    position = kwargs.get("position")

    client_file = path
    client_dir = output.get_client_name(client_file)
    client_output = f"../output/{client_dir}"

    this_dir = os.path.dirname(os.path.realpath(__file__))

    if kwargs.get("alignment"):
        logger.info("Running alignment check & correction algorithm...")
        subprocess.run(
            [
                "python",
                f"{this_dir}/alignment.py",
                client_file,
                "--position",
                position,
                "--directory",
                kwargs.get("directory"),
            ],
            check=True,
        )
        logger.info("Alignment check & correction complete.")

    # if kwargs.get("minimesh"):
    #     logger.info("Running minimesh construction algorithm...")
    #     subprocess.run(
    #         [
    #             "python",
    #             f"{this_dir}/minimesh.py",
    #             client_output,
    #             "--position",
    #             position,
    #         ],
    #         check=True,
    #     )
    #     logger.info("Minimesh construction algorithm complete.")
    #
    # if kwargs.get("vectors"):
    #     logger.info("Running vector construction algorithm...")
    #     subprocess.run(
    #         ["python", f"{this_dir}/vector_output.py", path, position],
    #         check=True,
    #     )
    #     logger.info("Vector construction algorithm complete.")

    logger.info(f"Files output to directory {client_output}")


@click.command()
@click.argument("path")
@click.option(
    "--position",
    default=None,
    type=click.Choice(["upper", "haut", "lower", "bas", None]),
    help="Is this the upper or lower jaw",
)
@click.option(
    "--directory",
    default=OUTPUT_DIR,
    help="Which directory to save models",
    type=str,
)
@click.option("--alignment/--no-alignment", default=False)
# @click.option("--minimesh/--no-minimesh", default=False)
# @click.option("--vectors/--no-vectors", default=False)
@logger.catch
def do_all_cli(path, **kwargs):
    # Automatically detect if this is an upper or lower
    # client jaw file. Easier cli processing
    position = kwargs.get("position")
    if position is None:
        if "haut" in path:
            kwargs["position"] = "upper"
        elif "bas" in path:
            kwargs["position"] = "lower"

    try:
        do_all(path, **kwargs)
    except subprocess.CalledProcessError:
        print("Error: subprocess returned a non-zero exit code")
        sys.exit(1)


if __name__ == "__main__":
    do_all_cli()

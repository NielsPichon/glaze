import argparse
import json
import pathlib
from typing import Dict, Sequence, Set


from PIL import Image
from loguru import logger
import numpy as np
import requests
import tqdm
from tqdm.contrib import concurrent



def retrieve_all_ingredients(recipes_dir: pathlib.Path) -> Set[str]:
    """
    Retrieve all ingredients from the recipes in the given directory.

    Args:
       recipes_dir (pathlib.Path): The directory containing the recipes.

    Returns:
        A set of all ingredients.
    """
    logger.info('Retrieving all available ingredients')
    ingredients = set()
    for recipe_file in tqdm.tqdm(list(recipes_dir.iterdir())):
        with open(recipe_file, "r") as f:
            recipes = json.load(f)

        for recipe in recipes:
            if recipe["ingredients"] is None or len(recipe["ingredients"]) == 0:
                continue
            ingredients.update(ingredient["name"] for ingredient
                               in recipe["ingredients"])

    return ingredients


def retrieve_all_atmosphere(recipes_dir: pathlib.Path) -> Set[str]:
    """
    Retrieve all atmospheres from the recipes in the given directory.

    Args:
       recipes_dir (pathlib.Path): The directory containing the recipes.

    Returns:
        A set of all atmospheres.
    """
    logger.info('Retrieving all available atmospheres')
    atmospheres = set()
    for recipe_file in tqdm.tqdm(list(recipes_dir.iterdir())):
        with open(recipe_file, "r") as f:
            recipes = json.load(f)

        for recipe in recipes:
            if (recipe["atmosphere"] is None
                    or recipe["atmosphere"] in ["?", ""]):
                continue
            atmospheres.update(recipe["atmosphere"].split(', '))

    return atmospheres


def load_image(image_url: str) -> np.ndarray:
    """
    Load an image from the given path.

    Args:
        image_url (str): The url to download image from.

    Returns:
        The loaded image.
    """
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    return np.array(image)


def process_recipe_file(file_path: str,
                        ingredients_mapping: Dict[str, int],
                        atmosphere_mapping: Dict[str, int],
                        output_dir: pathlib.Path):
    """Converts all the recipes inside a file to a dict of np arrays.

    Args:
        file_path (str): Path to the json file contianing the recipes.
        ingredients_mapping (Dict[str, int]): Mapping from ingredient to the
            index in the ingredients array.
        atmosphere_mapping (Dict[str, int]): Mapping from atmosphere to the
            index in the atmosphere array.
        output_dir (pathlib.Path): directory where to save the data points.
    """
    with open(file_path, "r") as f:
        recipes = json.load(f)

    for i, recipe in enumerate(tqdm.tqdm(recipes, leave=False)):
        file_name = output_dir / f"{file_path.stem}_{i}.npy"
        if file_name.exists():
            continue

        if recipe["ingredients"] is None or len(recipe["ingredients"]) == 0:
            continue

        datum = {}
        datum['amount'] = np.zeros(len(ingredients_mapping))
        datum['extras'] = np.zeros(len(ingredients_mapping))
        for ingredient in recipe["ingredients"]:
            amount = ingredient["amount"] / 100
            idx = ingredients_mapping[ingredient["name"]]
            key = "amount" if ingredient["base"] else "extras"
            datum[key][idx] = amount

        cone  = recipe["cone"]
        if cone != "?":
            if "-" in cone:
                cone = cone.split("-")[0]
            cone = cone.replace("\u00bd", ".5")
            cone = float(cone)
            datum["cone"] = cone

        atmosphere = recipe["atmosphere"]
        if atmosphere and atmosphere != "?":
            datum["atmosphere"] = np.zeros(len(atmosphere_mapping))
            atmospheres = atmosphere.split(', ')
            for atm in atmospheres:
                datum["atmosphere"][atmosphere_mapping[atm]] = 1

        try:
            datum['input'] = load_image(recipe["image_url"])
        except Exception as e:
            logger.error("Error loading image for recipe "
                         f"{recipe['name']}: {e}")
            continue

        np.save(file_name, datum)


def list_to_idx_dict(items: Sequence[str]) -> Dict[str, int]:
    """
    Create a dictionary mapping items to indices.

    Args:
        items (Sequence[str]): The items to map.

    Returns:
        A dictionary mapping items to indices.
    """
    return {item: idx for idx, item in enumerate(items)}


def prepare_dataset(recipes_dir: pathlib.Path,
                    output_dir: pathlib.Path,
                    max_workers: int = 8):
    """
    Prepare the dataset for training.

    Args:
        recipes_dir (pathlib.Path): The directory containing the recipes.
        output_dir (pathlib.Path): The directory where to save the dataset.
        max_workers (int): The number of workers to use for processing. Defaults
            to 8.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ingredients = retrieve_all_ingredients(recipes_dir)
    ingredients = list(ingredients)
    logger.info(f"Found {len(ingredients)} ingredients")

    with open(output_dir / "ingredients.txt", "w") as f:
        f.write("\n".join(ingredients))

    ingredient_to_idx = list_to_idx_dict(ingredients)

    atmospheres = list(retrieve_all_atmosphere(recipes_dir))
    atmospheres_to_idx = list_to_idx_dict(atmospheres)
    logger.info(f"Found the following atmospheres: {atmospheres}")

    with open(output_dir / "atmosphere.txt", "w") as f:
        f.write("\n".join(atmospheres))

    logger.info("Processing recipes")
    recipes = list(recipes_dir.iterdir())
    concurrent.thread_map(process_recipe_file,
                          recipes,
                          [ingredient_to_idx] * len(recipes),
                          [atmospheres_to_idx] * len(recipes),
                          [output_dir] * len(recipes),
                          max_workers=max_workers)


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", type=pathlib.Path, default="recipes")
    parser.add_argument("--out", "-o", type=pathlib.Path, default="data")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    prepare_dataset(args.recipes, args.out, args.max_workers)

import atexit
import json
import os
import pathlib
from typing import List

from loguru import logger
from pydantic import BaseModel
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tqdm


class Ingredient(BaseModel):
    name: str
    url: str
    amount: float
    base: bool = False


class Recipe(BaseModel):
    name: str
    url: str
    image_url: str
    tag: str
    cone: str
    atmosphere: str = ''
    surface: str = ''
    transparency: str = ''
    ingredients: List[Ingredient] = None



def get_root_url(page) -> str:
    return f'https://glazy.org/search?base_type=460&p={page}&photo=true'

class Browser:
    def __init__(self):
        os.environ['MOZ_HEADLESS'] = '1'
        self.browser = webdriver.Firefox()
        atexit.register(self.browser.close)

    def go_to_page(self, page, condition=(By.ID, 'glazy-app-div')):
        self.browser.get(page)
        WebDriverWait(self.browser, 10).until(
            EC.presence_of_all_elements_located(condition)
        )

    @property
    def page_source(self):
        return self.browser.page_source

    def find_elements(self, by, value):
        return self.browser.find_elements(by, value)


def scrape_root_page(page_index, browser) -> List[Recipe]:
    try:
        browser.go_to_page(get_root_url(page_index), (By.CSS_SELECTOR, 'div.bg-white.dark\\:bg-gray-800.shadow.rounded-md.sm\\:rounded-lg'))
        cards = browser.find_elements(By.CSS_SELECTOR, 'div.bg-white.dark\\:bg-gray-800.shadow.rounded-md.sm\\:rounded-lg')
    except Exception as e:
        logger.error(e)
        return  []

    recipes = []
    for card in tqdm.tqdm(cards, leave=False):
        try:
            try:
                type_tag = card.find_element(By.CSS_SELECTOR, 'div.absolute.bottom-1.right-1')
                type_tag = type_tag.find_element(By.TAG_NAME, 'span').text
                if type_tag.lower == 'discontinued':
                    continue
            except Exception as e:
                # it is very possible that the tag is not present
                type_tag = ''
                pass

            title_section = card.find_element(By.CSS_SELECTOR, 'h4')
            link = title_section.find_element(By.TAG_NAME, 'a')

            recipe = Recipe(
                name=link.find_element(By.TAG_NAME, 'span').text,
                url=link.get_attribute('href'),
                image_url=card.find_element(By.CSS_SELECTOR, 'img').get_attribute('src'),
                tag=type_tag,
                cone=title_section.find_elements(By.TAG_NAME, 'span')[-1].text[1:],
            )
            recipes.append(recipe)
        except Exception as e:
            logger.error(e)
            continue

    return recipes


def scrape_recipe(recipe: Recipe, browser) -> Recipe:
    try:
        browser.go_to_page(recipe.url, (By.TAG_NAME, 'table'))
    except Exception as e:
        logger.error(e)
        return recipe
    try:
        atmosphere_card = browser.find_elements(By.CSS_SELECTOR, 'div.flex-shrink-0.w-full.sm\\:w-auto')[0]
        atmosphere = atmosphere_card.find_element(By.CSS_SELECTOR, 'span.text-sm').text
        recipe.atmosphere = atmosphere
    except Exception as e:
        logger.error(f"atmosphere{recipe.url}: {e}")

    try:
        selectors = browser.find_elements(By.CSS_SELECTOR, 'div.flex-shrink-0.rounded-md.bg-gray-100.dark\\:bg-gray-750.px-3.py-1')
        recipe.surface = selectors[1].find_elements(By.TAG_NAME, 'div')[-1].text
        recipe.transparency = selectors[2].find_elements(By.TAG_NAME, 'div')[-1].text
    except Exception as e:
        logger.error(f"selectors {recipe.url}: {e}")

    try:
        ingredients = (browser.find_elements(By.CSS_SELECTOR, 'div.mt-4')[0]
                       .find_element(By.TAG_NAME, 'table')
                       .find_element(By.TAG_NAME, 'tbody')
                       .find_elements(By.TAG_NAME, 'tr'))
    except Exception as e:
        logger.error(f"ingredients {recipe.url}: {e}")

    recipe.ingredients = []
    for ingredient in ingredients:
        try:
            ingredient_link = ingredient.find_element(By.TAG_NAME, 'a')
            # TODO: fix this
            name = ingredient_link.text or ingredient_link.find_element(By.TAG_NAME, 'img').get_attribute('alt')
            ref = ingredient_link.get_attribute('href')
            amount = ingredient.find_elements(By.TAG_NAME, 'td')[-1].text
            recipe.ingredients.append(
                Ingredient(name=name, url=ref, amount=amount))
            try:
                ingredient.find_element(By.TAG_NAME, 'svg')
            except Exception as e:
                recipe.ingredients[-1].base = True
        except Exception as e:
            continue


def dump_recipes(recipes: List[Recipe], file_path: str) -> None:
    json_recipes = [recipe.model_dump() for recipe in recipes]
    with open(file_path, 'w') as f:
        json.dump(json_recipes, f, indent=2)


def scrape_glazy():
    recipes_dir = pathlib.Path('recipes')
    recipes_dir.mkdir(exist_ok=True, parents=True)

    browser = Browser()

    progress_bar = tqdm.tqdm(range(1, 425))
    for i in progress_bar:
        recipes_path = recipes_dir / f'{i}.json'
        progress_bar.set_description(f'Scanning root page')
        recipes = scrape_root_page(i, browser)
        dump_recipes(recipes, recipes_path)
        progress_bar.set_description(f'Scanning recipes')
        for recipe in tqdm.tqdm(recipes):
            scrape_recipe(recipe, browser)
            dump_recipes(recipes, recipes_path)


if __name__ == '__main__':
    scrape_glazy()

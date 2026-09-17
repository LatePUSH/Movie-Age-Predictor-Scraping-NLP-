# Projet scraper 2024 - Ingénierie des langues 

import requests
from bs4 import BeautifulSoup
import time
import json

def scrape_allocine(base_url, max_count=5000): # Nombre de films à scraper par genre
    films = []

# Headers pour éviter de se faire bloquer
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    page = 1
    while len(films) < max_count:
        url = f"{base_url}?page={page}" 
        print(f"Fetching URL: {url}") # Récupération de l'url pour le début du scraping
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch URL: {url} with status code: {response.status_code}")
            break
        soup = BeautifulSoup(response.text, 'html.parser')

        movie_items = soup.find_all('div', class_='card entity-card entity-card-list cf')
        if not movie_items:
            print("Plus de films trouvés ni de fin de pages.")
            break
        for movie_item in movie_items:
            if len(films) >= max_count:
                break
            title_tag = movie_item.find('a', class_='meta-title-link')
            title = title_tag.text.strip() if title_tag else "Titre inconnu"
            synopsis_tag = movie_item.find('div', class_='content-txt')
            synopsis = synopsis_tag.text.strip() if synopsis_tag else "Synopsis non disponible"
            movie_url = "https://www.allocine.fr" + title_tag['href'] if title_tag and 'href' in title_tag.attrs else None

            # Récupération de la classification par âge à partir de la page d'un film 
            age = "Tranche d'âge non disponible"
            if movie_url:
                time.sleep(0.5)  # Ajout d'un délai pour éviter d'être bloqué
                movie_response = requests.get(movie_url, headers=headers)
                if movie_response.status_code == 200:
                    movie_soup = BeautifulSoup(movie_response.text, 'html.parser')
                    age_tag = movie_soup.find('div', class_='certificate')
                    age = age_tag.text.strip() if age_tag else "Tranche d'âge non disponible"

            films.append({'titre': title, 'synopsis': synopsis, 'age': age})
            current_progress = len(films) / max_count * 100  # Ajout d'un % pour se situer dans la progression du scraping
            print(f"film ajouté: {title} - progression: {current_progress:.2f}%")

        page += 1
        time.sleep(1)

    return films

# URLs spécifiques aux genres pour Allociné
horror_url = "https://www.allocine.fr/films/genre-13009/"
action_url = "https://www.allocine.fr/films/genre-13025/"
romance_url = "https://www.allocine.fr/films/genre-13024/"

# Exécution du scraping pour chaque genre
horror_movies = scrape_allocine(horror_url)
action_movies = scrape_allocine(action_url)
romance_movies = scrape_allocine(romance_url)

# Enregistrement des résultats dans des fichiers JSON
with open('horreur.json', 'w', encoding='utf-8') as f:
    json.dump(horror_movies, f, indent=4, ensure_ascii=False)
with open('action.json', 'w', encoding='utf-8') as f:
    json.dump(action_movies, f, indent=4, ensure_ascii=False)
with open('romance.json', 'w', encoding='utf-8') as f:
    json.dump(romance_movies, f, indent=4, ensure_ascii=False)

print(f"Total des films d'horreur scraped: {len(horror_movies)}")
print(f"Total des films d'action scraped: {len(action_movies)}")
print(f"Total des films romance scraped: {len(romance_movies)}")

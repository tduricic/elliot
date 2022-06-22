import requests
import datetime
import bz2
import shutil
from tqdm import tqdm
from pathlib import Path


def prepare_directories(datasets):
    directories_to_create = ['downloaded', 'raw', 'processed']
    for dataset in datasets:
        for directory in directories_to_create:
            dirname = '/'.join(['data', dataset, directory])
            Path(dirname).mkdir(parents=True, exist_ok=True)

def prepare_epinions():
    # Epinions
    # Fetching and and unpacking ratings...
    print("Preparing Epinions dataset...")
    epinions_ratings_url = "http://www.trustlet.org/datasets/downloaded_epinions/ratings_data.txt.bz2"
    print(f"Getting Epinions ratings from: {epinions_ratings_url}")
    response = requests.get(epinions_ratings_url)

    downloaded_filename = 'data/epinions/downloaded/ratings_data.txt.bz2'
    open(downloaded_filename, 'wb').write(response.content)

    unpacked_filename = 'data/epinions/raw/ratings_data.txt'
    with open(unpacked_filename, 'wb') as unpacked_file, bz2.BZ2File(downloaded_filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            unpacked_file.write(data)

    # Preprocessing ratings...
    print("Preprocessing ratings:")
    with open(unpacked_filename) as fr:
        processed_lines = []
        for line in tqdm(fr.readlines()[1:-1]):
            tokens = line.replace("\n", "").split()
            userId = tokens[0]
            itemId = tokens[1]
            rating = tokens[2]

            delimiter = "\t"
            processed_lines.append(delimiter.join([userId, itemId, rating]) + "\n")
        with open("data/epinions/processed/ratings.tsv", "w") as fp:
            fp.writelines(processed_lines)

    # Fetching and unpacking social edges...
    epinions_edges_url = "http://www.trustlet.org/datasets/downloaded_epinions/trust_data.txt.bz2"
    print(f"Getting Epinions edges from : {epinions_edges_url}")
    response = requests.get(epinions_edges_url)

    downloaded_filename = 'data/epinions/downloaded/trust_data.txt.bz2'
    open(downloaded_filename, 'wb').write(response.content)

    unpacked_filename = 'data/epinions/raw/trust_data.txt'
    with open(unpacked_filename, 'wb') as unpacked_file, bz2.BZ2File(downloaded_filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            unpacked_file.write(data)

    # Preprocessing social edges...
    print("Preprocessing social edges:")
    with open(unpacked_filename) as fr:
        processed_lines = []
        for line in tqdm(fr.readlines()[1:-1]):
            tokens = line.replace("\n", "").split(" ")
            userId1 = tokens[0]
            userId2 = tokens[1]
            weight = tokens[2]

            delimiter = "\t"
            processed_lines.append(delimiter.join([userId1, userId2, weight]) + "\n")
        with open("data/epinions/processed/social_connections.tsv", "w") as fp:
            fp.writelines(processed_lines)
    print("Epinions dataset downloaded and preprocessed.")

def prepare_ciao():
    # Ciao
    print("Preparing Ciao dataset...")
    ciao_dataset_url = "https://guoguibing.github.io/librec/datasets/CiaoDVD.zip"
    print(f"Getting Ciao dataset from: {ciao_dataset_url}")
    response = requests.get(ciao_dataset_url)
    downloaded_filename = 'data/ciao/downloaded/CiaoDVD.zip'
    open(downloaded_filename, 'wb').write(response.content)

    targetdir = "data/ciao/raw"
    shutil.unpack_archive(downloaded_filename, targetdir)

    # Preprocessing ratings...
    print("Preprocessing ratings:")
    with open("data/ciao/raw/movie-ratings.txt") as fr:
        lines = []
        for line in tqdm(fr.readlines()):
            tokens = line.split(",")
            userId = tokens[0]
            itemId = tokens[1]
            rating = tokens[4]
            date = tokens[5]
            timestamp = int(datetime.datetime.strptime(date.strip(), '%Y-%m-%d').strftime("%s"))

            delimiter = "\t"
            lines.append(delimiter.join([userId, itemId, rating, str(timestamp)]) + "\n")
        with open("data/ciao/processed/ratings.tsv", "w") as fp:
            fp.writelines(lines)

    # Preprocessing social edges...
    print("Preprocessing social edges:")
    with open("data/ciao/raw/trusts.txt") as fr:
        lines = []
        for line in tqdm(fr.readlines()):
            tokens = line.split(",")
            userId1 = tokens[0]
            userId2 = tokens[1]
            weight = tokens[2]

            delimiter = "\t"
            lines.append(delimiter.join([userId1, userId2, weight]))
        with open("data/ciao/processed/social_connections.tsv", "w") as fp:
            fp.writelines(lines)
    print("Ciao dataset downloaded and preprocessed.")

def prepare_douban():
    # Douban
    print("Preparing Douban dataset...")
    # You need to download the dataset archive from here and put it in the 'data/douban/downloaded' directory
    douban_dataset_url = "https://www.dropbox.com/s/u2ejjezjk08lz1o/Douban.tar.gz?dl=0"
    downloaded_filename = 'data/douban/downloaded/Douban.tar.gz'

    targetdir = "data/douban/raw"
    shutil.unpack_archive(downloaded_filename, targetdir)

    # Preprocessing book ratings...
    print("Preprocessing book ratings:")
    with open("data/douban/raw/Douban/book/douban_book.tsv") as fr:
        lines = []
        for line in tqdm(fr.readlines()[1:]):
            tokens = line.split()
            userId = tokens[0]
            itemId = tokens[1]
            rating = tokens[2]
            # We skip negative ratings
            if int(rating) == -1:
                continue
            timestamp = int(float(tokens[3]))

            delimiter = "\t"
            lines.append(delimiter.join([userId, itemId, rating, str(timestamp)]) + "\n")
        with open("data/douban/processed/book_ratings.tsv", "w") as fp:
            fp.writelines(lines)

    # Preprocessing movie ratings...
    print("Preprocessing movie ratings:")
    with open("data/douban/raw/Douban/movie/douban_movie.tsv") as fr:
        lines = []
        for line in tqdm(fr.readlines()[1:]):
            tokens = line.split()
            userId = tokens[0]
            itemId = tokens[1]
            rating = tokens[2]
            # We skip negative ratings
            if int(rating) == -1:
                continue
            timestamp = int(float(tokens[3]))

            delimiter = "\t"
            lines.append(delimiter.join([userId, itemId, rating, str(timestamp)]) + "\n")
        with open("data/douban/processed/movie_ratings.tsv", "w") as fp:
            fp.writelines(lines)

    # Preprocessing music ratings...
    print("Preprocessing music ratings:")
    with open("data/douban/raw/Douban/music/douban_music.tsv") as fr:
        lines = []
        for line in tqdm(fr.readlines()[1:]):
            tokens = line.split()
            userId = tokens[0]
            itemId = tokens[1]
            rating = tokens[2]
            # We skip negative ratings
            if int(rating) == -1:
                continue
            timestamp = int(float(tokens[3]))

            delimiter = "\t"
            lines.append(delimiter.join([userId, itemId, rating, str(timestamp)]) + "\n")
        with open("data/douban/processed/music_ratings.tsv", "w") as fp:
            fp.writelines(lines)

    # Preprocessing social edges...
    print("Preprocessing social edges:")
    with open("data/douban/raw/Douban/socialnet/socialnet.tsv") as fr:
        lines = []
        for line in tqdm(fr.readlines()[1:]):
            tokens = line.split()
            userId1 = tokens[0]
            userId2 = tokens[1]
            weight = tokens[2]

            delimiter = "\t"
            lines.append(delimiter.join([userId1, userId2, str(int(float(weight)))]) + "\n")
        with open("data/douban/processed/social_connections.tsv", "w") as fp:
            fp.writelines(lines)
    print("Ciao dataset downloaded and preprocessed.")

def prepare_lastfm():
    # LastFM
    print("Preparing LastFM dataset...")
    lastfm_dataset_url = "https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip"
    print(f"Getting LastFM dataset from : {lastfm_dataset_url}")
    response = requests.get(lastfm_dataset_url)
    downloaded_filename = 'data/lastfm/downloaded/hetrec2011-lastfm-2k.zip'
    open(downloaded_filename, 'wb').write(response.content)

    targetdir = "data/lastfm/raw"
    shutil.unpack_archive(downloaded_filename, targetdir)

    # Preprocessing music ratings...
    print("Preprocessing music ratings:")
    with open("data/lastfm/raw/user_artists.dat") as fr:
        lines = []
        for line in tqdm(fr.readlines()[1:]):
            tokens = line.split()
            userId = tokens[0]
            itemId = tokens[1]
            # TODO maybe we should already normalize ratings here somehow?
            rating = tokens[2]

            delimiter = "\t"
            lines.append(delimiter.join([userId, itemId, rating]) + "\n")
        with open("data/lastfm/processed/ratings.tsv", "w") as fp:
            fp.writelines(lines)

    # Preprocessing social edges...
    print("Preprocessing social edges:")
    with open("data/lastfm/raw/user_friends.dat") as fr:
        lines = []
        for line in tqdm(fr.readlines()[1:]):
            tokens = line.split()
            userId1 = tokens[0]
            userId2 = tokens[1]
            # Graph is undirected and unweighted
            weight = '1'

            delimiter = "\t"
            lines.append(delimiter.join([userId1, userId2, weight]) + "\n")
        with open("data/lastfm/processed/social_connections.tsv", "w") as fp:
            fp.writelines(lines)
    print("LastFM dataset downloaded and preprocessed.")

def prepare_filmtrust():
    # Filmtrust
    print("Preparing Filmtrust dataset...")
    filmtrust_dataset_url = "https://guoguibing.github.io/librec/datasets/filmtrust.zip"
    print(f"Getting Filmtrust dataset from: {filmtrust_dataset_url}")
    response = requests.get(filmtrust_dataset_url)
    downloaded_filename = 'data/filmtrust/downloaded/filmtrust.zip'
    open(downloaded_filename, 'wb').write(response.content)

    targetdir = "data/filmtrust/raw"
    shutil.unpack_archive(downloaded_filename, targetdir)

    # Preprocessing ratings...
    print("Preprocessing ratings:")
    with open("data/filmtrust/raw/ratings.txt") as fr:
        lines = []
        for line in tqdm(fr.readlines()):
            tokens = line.split()
            userId = tokens[0]
            itemId = tokens[1]
            rating = tokens[2]

            delimiter = "\t"
            lines.append(delimiter.join([userId, itemId, rating]) + "\n")
        with open("data/filmtrust/processed/ratings.tsv", "w") as fp:
            fp.writelines(lines)

    # Preprocessing social edges...
    print("Preprocessing social edges:")
    with open("data/filmtrust/raw/trust.txt") as fr:
        lines = []
        for line in tqdm(fr.readlines()):
            tokens = line.split()
            userId1 = tokens[0]
            userId2 = tokens[1]
            weight = tokens[2]

            delimiter = "\t"
            lines.append(delimiter.join([userId1, userId2, weight]) + "\n")
        with open("data/filmtrust/processed/social_connections.tsv", "w") as fp:
            fp.writelines(lines)
    print("Filmtrust dataset downloaded and preprocessed.")

def main():
    # datasets = ['epinions', 'ciao', 'douban', 'lastfm', 'filmtrust']
    datasets = ['filmtrust']
    prepare_directories(datasets)

    data_prep_dict = {"epinions": prepare_epinions,
                      "ciao" : prepare_ciao,
                      "douban" : prepare_douban,
                      "lastfm":prepare_lastfm,
                      "filmtrust":prepare_filmtrust}

    for dataset in datasets:
        data_prep_dict[dataset]()

if __name__ == "__main__":
    main()

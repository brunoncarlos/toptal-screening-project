import requests

def download_from_github_repo(path, save_as=None):
    url = f"https://github.com/brunoncarlos/toptal-screening-project/raw/main/{path}"
    r = requests.get(url)

    if r.status_code != 200:
        raise Exception(f"Error descargando {path}: {r.status_code}")

    if save_as is None:
        save_as = path.split("/")[-1]

    with open(save_as, "wb") as f:
        f.write(r.content)

    print(f"✔ Archivo descargado: {save_as}")

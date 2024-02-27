from pathlib import Path

from company import Company


def parse_company_file_from_path(file_path: str) -> Company:
    with open(file_path, "r") as fp:
        company_name = fp.readline().strip()
        _ = fp.readline()  # number of available datapoints; discarded
        line = fp.readline()
        prices = []
        while line:
            _, price = line.split()  # index of the price is discarded
            prices.append(float(price))
            line = fp.readline()
    return Company(company_name, prices)


def load_all_companies_from_dir(directory_path: str) -> list[Company]:
    """Load all files in a directory that contain info about stock prices
    into a list of Companies. Assumes that all .txt files in the directory
    are valid
    """
    files = Path(directory_path).glob("*.txt")
    return [parse_company_file_from_path(file) for file in files]


if __name__ == "__main__":
    example_company = parse_company_file_from_path("./data/Bundle1/WorldNow_Part1.txt")
    # print(example_company)
    companies = load_all_companies_from_dir("./data/Bundle1")
    for c in companies:
        print(c)

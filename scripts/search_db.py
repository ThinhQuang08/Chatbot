from services.search_service import search_destinations


def print_results(results):
    if not results:
        print("\nNo results found.\n")
        return

    print("\nResults:\n")

    for r in results:
        print(f"Location: {r['location']}")
        print(f"Season: {r['season']}")
        print(f"Description: {r['description']}")
        print(f"Activities: {r['activities']}")
        print(f"Cost: {r['cost']}")
        print(f"Price level: {r['price_level']}")
        print("-" * 40)


if __name__ == "__main__":
    query = input("Enter location, activity, season or price: ")

    results = search_destinations(query)

    print_results(results)

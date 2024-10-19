import os
import sys
import requests
from googlesearch import search
from bs4 import BeautifulSoup

def google_search_and_save(save_directory, search_string, num_results=100):
    # Create the save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Perform the Google search
    search_results = search(search_string, num_results=num_results)

    for i, url in enumerate(search_results, 1):
        try:
            # Send a GET request to the URL
            response = requests.get(url, timeout=10)
            
            # Check if the request was successful
            if response.status_code == 200:
                # Parse the HTML content
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Generate a filename for the HTML file
                filename = f"result_{i}.html"
                filepath = os.path.join(save_directory, filename)
                
                # Save the HTML content to a file
                with open(filepath, 'w', encoding='utf-8') as file:
                    file.write(str(soup))
                
                print(f"Saved: {filename}")
            else:
                print(f"Failed to retrieve: {url}")
        
        except Exception as e:
            print(f"Error processing {url}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <save_directory> <search_string>")
        sys.exit(1)

    save_directory = sys.argv[1]
    search_string = sys.argv[2]

    google_search_and_save(save_directory, search_string)
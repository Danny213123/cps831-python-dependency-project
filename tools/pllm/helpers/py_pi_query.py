# Methods for querying PyPi directly
# Used for finding rough versions
import json
import os
import re
from datetime import datetime
from types import SimpleNamespace

import requests

try:
    from pypi_json import PyPIJSON  # type: ignore
except Exception:
    PyPIJSON = None

from helpers.github_cruiser_core import GithubCruiserCore
from helpers.deps_scraper import DepsScraper

class PyPIQuery:
    ###
    # For now we use GithubCruiserCore for certain helper functions
    ###
    def __init__(self, logging=False, base_modules="./modules") -> None:
        self.date_format = '%Y-%m-%d'
        self.output_date_format = '%b %d %Y'
        self.logging = False
        self.ghc = GithubCruiserCore(logging=False)
        self.deps = DepsScraper(logging=logging)
        self.python_versions = self.ghc.load_json_from_file("helpers/ref_files/python_versions.json")
        os.makedirs(base_modules, exist_ok=True)
        self.base_modules = base_modules

    def check_format(self, python_version):
        python_version = python_version.replace('+', '')
        split_version = python_version.split('.')
        if len(split_version) == 1:
            return f"{split_version[0]}.7"
        else:
            checked_version = f"{split_version[0]}.{split_version[1] if split_version[1] != 'x' else '7'}"
            return checked_version
    
    
    def read_module_file(self, module, python_version):
        file = f"{self.base_modules}/{module}_{python_version}.txt"
        if os.path.isfile(file):
            with open(file, 'r') as file:
                data = file.read()
            return data
        else:
            module_details = {'python_version': python_version, 'python_modules': [module]}
            out = self.get_module_specifics(module_details)
            if os.path.isfile(file):
                with open(file, 'r') as file:
                    data = file.read()
                return data
            else:
                return ''
    

    # Get a start and end date based on the Python versions we're using
    # This takes the Python version and creates a date range from release, through to the next version.
    # NOTE: DOUBLE CHECK THIS, POSSIBLE BAD RETURN IN CERTAIN CASES!
    def get_python_dates(self, python_version):
        checked_version = self.check_format(python_version)
        selected_idx = None

        # Loop through the different python versions
        for idx, x in enumerate(self.python_versions):
            if checked_version == x['cycle']:
                selected_idx = idx
                break

        if selected_idx is None:
            fallback_cycle = "3.8"
            for idx, x in enumerate(self.python_versions):
                if x['cycle'] == fallback_cycle:
                    selected_idx = idx
                    checked_version = fallback_cycle
                    break
            if selected_idx is None:
                selected_idx = 0
                checked_version = self.python_versions[0]['cycle']

        current = self.python_versions[selected_idx]
        current_date = datetime.strptime(current['releaseDate'], self.date_format).date()
        if selected_idx > 0:
            next_date = datetime.strptime(self.python_versions[selected_idx - 1]['releaseDate'], self.date_format).date()
        else:
            next_date = datetime.now().date()
        return current_date, next_date, checked_version

    # Get a range of Python versions based on the given version
    # For example if we give Python 3.7 it will return [3.5, 3.6, 3.7, 3.8, 3.9]
    def get_python_range(self, python_version, pyrange=2):
        checked_version = self.check_format(python_version)
        selected_python = []

        try:
            cycles = [entry["cycle"] for entry in self.python_versions]
            if checked_version not in cycles:
                checked_version = "3.8" if "3.8" in cycles else cycles[0]

            target = cycles.index(checked_version)
            start_index = max(0, target - pyrange)
            end_index = min(len(cycles), target + pyrange + 1)

            # Try to keep the requested window size when close to boundaries.
            target_size = (pyrange * 2) + 1
            while (end_index - start_index) < target_size and start_index > 0:
                start_index -= 1
            while (end_index - start_index) < target_size and end_index < len(cycles):
                end_index += 1

            selected_python = cycles[start_index:end_index]
        except Exception as e:
            print(f"Unable to get Python version: {e}")

        if len(selected_python) <= 0:
            selected_python = ["3.8"]

        if self.logging:
            print(selected_python)
        return selected_python

    
    # Checks the modules to ensure they look correct
    # This ensures there's no weird formatting or the model went awry
    def check_modules(self, modules):
        f = open('./helpers/ref_files/module_link.json')
        known_modules = json.load(f)
        module_list = {}

        for module in modules:
            if '.' in module:
                module = module.split('.')[0]
            
            module_version = modules[module]
            if module in known_modules:
                module_list[known_modules[module.lower()]['ref']] = module_version
            else:
                module_list[module.lower()] = module_version

        return module_list            

    # Uses a json list to check if a module has a different name
    # Loops through the modules we're looking for, along with the known list of name variants
    # Creates a new array of module names
    def check_module_name(self, module_name):
        # module_name = ['jinja2', 'os', 'json', 'logging', 're', 'hashlib', 'hmac', 'random', 'string', 'time', 'google.appengine.ext', 'google.appengine.api', 'blog_main', 'webapp2', 'google.appengine.ext', 'datetime', 'logging', 'json']
        f = open('./helpers/ref_files/module_link.json')
        known_modules = json.load(f)
        module_list = []
        if type(module_name) == str:
            module_name = [module_name]

        for module in module_name:
            if '.' in module:
                module = module.split('.')[0]
            
            module = module.replace(';', '').replace(',', '')
            
            if module.lower() in known_modules:
                module_list.append(known_modules[module.lower()]['ref'])
            else:
                module_list.append(module.lower())

        module_list = self.deps.clean_deps(module_list)

        return module_list


    # Calls the PyPIJSON module to get a list of all a modules versions
    # Returns the request meta data
    def query_module(self, module_name):
        try:
            if PyPIJSON is not None:
                with PyPIJSON() as client:
                    return client.get_metadata(module_name)

            response = requests.get(f"https://pypi.org/pypi/{module_name}/json", timeout=10)
            if response.status_code != 200:
                return None
            payload = response.json()
            releases = payload.get("releases", {})
            if not isinstance(releases, dict):
                return None
            # Keep the existing call-sites unchanged by returning a simple object with a .releases attribute.
            return SimpleNamespace(releases=releases)
        except Exception as e:
            return None


    def find_modules(self, module_name, start_date, end_date, python_version):
        dpq = self.query_module(module_name)
        stored = []

        if not dpq: return stored

        modules_releases = dpq.releases
        latest_release = {'version': '', 'date': datetime.strptime('1981-10-02', self.date_format).date()}

        # If the module has 5 or less releases, then we take all regardless of date
        small_repo =  len(modules_releases) <= 5

        for ele in modules_releases:
            if self.logging: print(ele)
            if len(modules_releases) > 0:
                release = modules_releases[ele]
                if len(release) > 0:
                    store = None
                    for release_details in release:
                        if not store:
                            if not release_details['yanked']:
                                upload_time = datetime.strptime(release_details['upload_time'].split('T')[0], '%Y-%m-%d').date()
                                store = {}
                                
                                # Start block!
                                # If the releases in the module are less than equal to 5 then we store them all
                                if small_repo:
                                    store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}

                                # Typical block
                                # Store anything that's within our timeframe
                                if upload_time >= start_date and upload_time <= end_date:
                                    store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                                elif self.get_version_from_code(release_details['python_version']) == python_version:
                                    store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                                    if self.logging: print(f"{module_name} with {release_details['python_version']}: found outside date {start_date} -> {end_date}: {store}")
                                elif 'py2' in release_details['python_version'] and '2.' in python_version:
                                    store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                                elif 'py3' in release_details['python_version'] and '3.' in python_version:
                                    store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                                # elif upload_time > end_date and len(stored) <= 0:
                                #     store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}
                                elif 'source' in release_details['python_version'] and len(stored) <= 20:
                                    store = {'version': ele, 'date': upload_time.strftime(self.output_date_format)}

                                # Make sure we always store the latest version
                                if upload_time >= latest_release['date']:
                                    latest_release = {'version': ele, 'date': upload_time}

                    if store: stored.append(store)
    
        if self.logging: print(f"start date: {start_date} | end date: {end_date}")
        
        if len(stored) == 0:
            latest_release['date'] = latest_release['date'].strftime(self.output_date_format)
            stored.append(latest_release)
        
        return stored

    def get_module_specifics(self, module_details={}):
        if self.logging: print(module_details)
        start_date, end_date, python_version = self.get_python_dates(module_details['python_version'])
        
        # Update the Python modules with exact install names
        python_modules = self.check_module_name(module_details['python_modules'])
        # Filter further through PyPi to create a curated list
        modified_modules = []


        # Function to split version strings and convert each part to an integer
        def version_key(version):
            # Split by '.' and convert numeric parts to integers. Handle non-numeric parts (e.g., '4.0.30b1') by keeping them as strings.
            return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', version)]


        for dep in python_modules:
            modules = self.find_modules(dep, start_date, end_date, python_version)
            module_versions = []
            if len(modules) > 0:
                # module_details = f"Module name: {dep}, Python version: {python_version}, Module versions: ["
                # module_details = f"Module versions: ["
                for idx, module in enumerate(modules):
                    # if idx > 0: module_details += ", "
                    # module_details += f"{module['version']} ({module['date']})"
                    # module_details += f"{module['version']}"
                    module_versions.append(module['version'])
                # module_details += "]"
                # if self.logging: print(module_details)
            
            modified_modules.append(dep)
            module_versions.sort(key=version_key)
            with open(f"{self.base_modules}/{dep}_{python_version}.txt", "w") as outfile:
                # outfile.write(f"Module versions: [")
                outfile.write(', '.join(module_versions))
                # outfile.write("]")
                # module_details = f"Module versions: ["
                
                # outfile.write(module_details)

        return modified_modules, python_version

    def get_version_from_code(self, python_code):
        version = ""

        if 'cp' in python_code:
            python_code = python_code[2:]
        else:
            return python_code
        
        for idx, c in enumerate(python_code):
            if idx < len(python_code)-1:
                version += f"{c}."
            else:
                version += c
        
        return version

def main():
    pp = PyPIQuery()
    sk = pp.query_module('clipboard')

    llm_details = {'python_version': '2.7', 'python_modules': ['clipboard']}
    pp.get_module_specifics(llm_details)

    print(sk)

if __name__ == "__main__":
    main()

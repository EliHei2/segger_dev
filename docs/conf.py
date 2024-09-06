import os
import sys
import sphinx_rtd_theme

# Add the project source directory to sys.path
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'Segger'
author = 'Elyas Heidari'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'autoapi.extension',   
    'sphinx_click',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxcontrib.bibtex',
    'myst_parser',
    'sphinx_book_theme',
    'myst_parser',  
    'sphinx_design',  # Add sphinx-design for UI components
    'sphinx_copybutton'
]

html_theme_options = {
    "logo": {
        "text": "Segger",  # Customize with your project's name
    },
    "external_links": [
        {"name": "GitHub", "url": "https://github.com/EliHei2/segger_dev"},
        {"name": "Documentation", "url": "https://github.com/EliHei2/segger_dev#readme"},
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/EliHei2/segger_dev",
            "icon": "fab fa-github-square",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "footer_items": ["copyright"],
    "show_nav_level": 2,  # Adjust how many levels of the sidebar are shown
    "navbar_align": "content",  # Center align the navigation bar
    "footer_items": ["copyright"],
}

# Static files (e.g., custom CSS, images)
html_static_path = ['_static']



bibtex_bibfiles = ['references.bib'] 

# Paths for templates and static files
templates_path = ['_templates']
html_static_path = ['_static']

# The master toctree document
master_doc = 'index'


# autoapi_type = 'python'
# autodoc_dirs = ['../src']       # Ensure this points to the folder containing your code
# # autoapi_root = 'api'            # Auto-generated API docs will be placed in docs/api
# autoapi_keep_files = True   
# # Autodoc settings
# autodoc_default_options = {
#     'members': True,
#     'undoc-members': True,
#     'private-members': True,
#     'show-inheritance': True,
# }

autoapi_type = 'python'
autoapi_dirs = ['../src']       # Ensure this points to the folder containing your code
autoapi_root = 'api'            # Auto-generated API docs will be placed in docs/api
autoapi_keep_files = True       # Keep the generated files
autoapi_options = [
    'members',
    'undoc-members',
    'private-members',
    'show-inheritance',
]



# Theming
# html_theme = 'sphinx_book_theme'
# html_theme_options = {
#     "repository_url": "https://github.com/EliHei2/segger_dev",
#     "use_repository_button": True,
#     "use_issues_button": True,
#     "use_edit_page_button": True,
#     "use_download_button": False,
#     "home_page_in_toc": True,
# }

# Source file suffixes
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Intersphinx configuration to link to other projects' documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# Copy button settings for code snippets
copybutton_prompt_text = ">>> "

# Path to your source files
sys.path.insert(0, os.path.abspath('../../src/segger'))

# Set the path to your CLI source files
sys.path.insert(0, os.path.abspath('../../src/cli'))
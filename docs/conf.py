project = "Tabulus"
author = "Tabulus contributors"

extensions = [
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "sphinx_book_theme"
html_title = "Tabulus Documentation"
html_logo = "_static/tabulus-logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/sciknoworg/tabulus",
    "use_repository_button": False,
    "use_issues_button": False,
    "use_edit_page_button": False,
    "show_navbar_depth": 2,
    "show_toc_level": 2,
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]

# Generative Modeling Tutorials - Documentation

This directory contains the MkDocs documentation for the generative modeling tutorials.

## Quick Start

### Local Preview

1. Install MkDocs Material:
   ```bash
   pip install mkdocs-material
   ```

2. Preview the docs locally:
   ```bash
   mkdocs serve
   ```

3. Open http://127.0.0.1:8000 in your browser

### Build Documentation

```bash
mkdocs build
```

This creates a `site/` directory with the static website.

## Deployment

### GitHub Pages

The documentation is automatically deployed to GitHub Pages when you push to the `main` branch.

**Setup:**

1. Go to your repository settings on GitHub
2. Navigate to Pages settings
3. Select "GitHub Actions" as the source
4. Push to `main` branch - the workflow will automatically deploy

The workflow is defined in `.github/workflows/deploy.yml`.

## Directory Structure

```
docs_package/
├── .github/workflows/
│   └── deploy.yml          # GitHub Actions workflow
├── docs/
│   ├── index.md            # Landing page
│   ├── installation.md     # Installation guide
│   ├── getting-started.md  # Setup guide
│   ├── faq.md             # FAQ
│   ├── troubleshooting.md # Troubleshooting
│   ├── tutorials/
│   │   ├── tutorial-1.md  # Tutorial 1: DDPM
│   │   ├── tutorial-2.md  # Tutorial 2: Flow Matching
│   │   ├── tutorial-3.md  # Tutorial 3: Coming Soon
│   │   ├── tutorial-4.md  # Tutorial 4: Coming Soon
│   │   └── tutorial-5.md  # Tutorial 5: Coming Soon
│   ├── javascripts/
│   │   └── mathjax.js     # MathJax config
│   └── stylesheets/
│       └── extra.css      # Custom styles
├── mkdocs.yml             # MkDocs configuration
└── README.md              # This file
```

## Configuration

The main configuration is in `mkdocs.yml`. Key sections:

- **theme**: Material theme settings
- **nav**: Navigation structure
- **markdown_extensions**: Enabled extensions
- **extra_javascript**: MathJax for LaTeX
- **extra_css**: Custom styling

## Customization

### Change Theme Colors

Edit `mkdocs.yml`:

```yaml
theme:
  palette:
    primary: indigo  # Change to: blue, red, green, etc.
    accent: indigo
```

### Add New Page

1. Create markdown file in `docs/`
2. Add to navigation in `mkdocs.yml`:

```yaml
nav:
  - New Page: newpage.md
```

### Update Repository URL

In `mkdocs.yml`, change:

```yaml
repo_name: yourusername/generative-tutorials
repo_url: https://github.com/yourusername/generative-tutorials
```

## Features

- **Material Theme**: Modern, responsive design
- **Math Support**: LaTeX via MathJax
- **Code Highlighting**: Syntax highlighting for all code blocks
- **Search**: Full-text search
- **Dark Mode**: Toggle between light/dark themes
- **Tabs**: Tabbed content for multiple platforms
- **Admonitions**: Info boxes, warnings, tips
- **Mermaid Diagrams**: Flow charts and diagrams

## Contributing

To contribute to the documentation:

1. Edit markdown files in `docs/`
2. Test locally with `mkdocs serve`
3. Submit a pull request

## License

Same license as the main repository (MIT).

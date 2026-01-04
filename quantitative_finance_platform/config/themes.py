"""
Goldman Sachs-inspired Theme Configuration
Professional color schemes and styling for the platform
"""

from config.settings import Config

class GoldmanSachsTheme:
    """Goldman Sachs visual theme configuration"""

    # Color Palette
    COLORS = {
        "primary": Config.GS_BLUE,
        "primary_dark": Config.GS_DARK_BLUE,
        "secondary": Config.GS_GOLD,
        "success": Config.GS_SUCCESS,
        "danger": Config.GS_DANGER,
        "light": Config.GS_LIGHT_GRAY,
        "dark": Config.GS_DARK_GRAY,
        "white": "#FFFFFF",
        "black": "#1A1A1A",
    }

    # Typography
    FONTS = {
        "heading": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "body": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "monospace": "Monaco, Consolas, monospace",
    }

    # Layout
    LAYOUT = {
        "sidebar_width": "280px",
        "max_content_width": "1400px",
        "card_radius": "8px",
        "shadow": "0 2px 8px rgba(0,0,0,0.1)",
        "shadow_hover": "0 4px 16px rgba(0,0,0,0.15)",
    }

    # Chart Colors (for plotly/matplotlib)
    CHART_COLORS = {
        "primary_line": COLORS["primary"],
        "secondary_line": COLORS["secondary"],
        "positive": COLORS["success"],
        "negative": COLORS["danger"],
        "neutral": COLORS["dark"],
        "background": COLORS["white"],
        "grid": "#E5E7EB",
    }

    # Color palettes for multi-series charts
    SEQUENTIAL_PALETTE = [
        "#0033A0",  # GS Blue
        "#0066CC",
        "#3399FF",
        "#66B2FF",
        "#99CCFF",
    ]

    DIVERGING_PALETTE = [
        "#E74C3C",  # Red (negative)
        "#F39C12",  # Orange
        "#95A5A6",  # Gray (neutral)
        "#2ECC71",  # Green (positive)
        "#0033A0",  # Blue (strong positive)
    ]

    @staticmethod
    def get_plotly_template():
        """Return custom Plotly template with GS theme"""
        template = {
            "layout": {
                "font": {"family": GoldmanSachsTheme.FONTS["body"], "color": GoldmanSachsTheme.COLORS["dark"]},
                "title": {"font": {"size": 24, "color": GoldmanSachsTheme.COLORS["primary"]}},
                "plot_bgcolor": GoldmanSachsTheme.COLORS["white"],
                "paper_bgcolor": GoldmanSachsTheme.COLORS["white"],
                "colorway": GoldmanSachsTheme.SEQUENTIAL_PALETTE,
                "hovermode": "x unified",
                "hoverlabel": {"bgcolor": GoldmanSachsTheme.COLORS["dark"], "font": {"color": GoldmanSachsTheme.COLORS["white"]}},
            }
        }
        return template

    @staticmethod
    def get_matplotlib_style():
        """Return matplotlib rcParams for GS theme"""
        style = {
            "figure.facecolor": GoldmanSachsTheme.COLORS["white"],
            "axes.facecolor": GoldmanSachsTheme.COLORS["white"],
            "axes.edgecolor": GoldmanSachsTheme.COLORS["dark"],
            "axes.labelcolor": GoldmanSachsTheme.COLORS["dark"],
            "axes.prop_cycle": f"cycler('color', {GoldmanSachsTheme.SEQUENTIAL_PALETTE})",
            "text.color": GoldmanSachsTheme.COLORS["dark"],
            "xtick.color": GoldmanSachsTheme.COLORS["dark"],
            "ytick.color": GoldmanSachsTheme.COLORS["dark"],
            "grid.color": GoldmanSachsTheme.CHART_COLORS["grid"],
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "font.family": "sans-serif",
            "font.size": 10,
        }
        return style

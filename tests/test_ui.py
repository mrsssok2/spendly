"""Shared page layout: the light/dark toggle and the no-flash theme script."""
import re

import pytest


@pytest.mark.parametrize("path", ["/", "/login", "/register"])
def test_theme_toggle_on_public_pages(client, path):
    page = client.get(path).get_data(as_text=True)
    assert 'id="theme-toggle"' in page


@pytest.mark.parametrize("path", ["/profile", "/analytics", "/expenses/add"])
def test_theme_toggle_on_logged_in_pages(client, user, path):
    page = client.get(path).get_data(as_text=True)
    assert 'id="theme-toggle"' in page


def test_theme_script_runs_in_head_before_the_stylesheet(client):
    page = client.get("/").get_data(as_text=True)
    head = re.search(r"<head>(.*?)</head>", page, re.DOTALL).group(1)
    assert "js/theme.js" in head                                   # applied before drawing: no white flash
    assert head.index("js/theme.js") < head.index("css/style.css")

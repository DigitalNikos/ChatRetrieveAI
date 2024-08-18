import streamlit as st

# HTML/CSS templates for tables
def get_table_style():
    return """
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
            color: black;
        }
        a {
            text-decoration: none;
            color: blue;
        }
    </style>
    """

# Utility functions
def hide_file_names():
    return """
    <style>
    div[data-testid="stFileUploaderFile"] {
        display: none;
    }
    </style>
    """

def add_divider(padding_top: int = 0, padding_bottom: int = 0):
    st.markdown(f"<div style='padding-top: {padding_top}px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 1rem;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding-bottom: {padding_bottom}px;'></div>", unsafe_allow_html=True)

def add_heading(text: str, level: int = 2, align: str = "left", padding_top: int = 0, padding_bottom: int = 0):
    st.markdown(f"<div style='padding-top: {padding_top}px;'></div>", unsafe_allow_html=True)
    st.markdown(f"<h{level} style='text-align: {align};'>{text}</h{level}>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding-bottom: {padding_bottom}px;'></div>", unsafe_allow_html=True)

def add_padding(padding_top: int = 0, padding_bottom: int = 0):
    st.markdown(f"<div style='padding-top: {padding_top}px; padding-bottom: {padding_bottom}px;'></div>", unsafe_allow_html=True)

def generate_table(headers, rows):
    table_style = get_table_style()
    table_markdown = table_style + "<table><thead><tr>"
    
    for header in headers:
        table_markdown += f"<th>{header}</th>"
    
    table_markdown += "</tr></thead><tbody>"
    
    for row in rows:
        table_markdown += "<tr>"
        for item in row:
            table_markdown += f"<td>{item}</td>"
        table_markdown += "</tr>"
    
    table_markdown += "</tbody></table>"
    return table_markdown

def display_table(headers, rows, max_visible_items=15):
    table_markdown = generate_table(headers, rows)
    
    # Set up the scrollable div if there are more rows than max_visible_items
    row_height = 40  # Approximate height for each row in pixels
    max_height = max_visible_items * row_height

    scrollable_div = f"""
    <style>
        .scrollable-table {{
            max-height: {max_height}px;
            overflow-y: auto;
            border: 1px solid #ddd;
        }}
    </style>
    <div class="scrollable-table">
        {table_markdown}
    </div>
    """
    st.markdown(scrollable_div, unsafe_allow_html=True)

def add_centered_heading_with_description(heading: str, description: str):
    """
    Add a centered heading and description text in the middle of the page.
    
    Args:
        heading (str): The heading text to display.
        description (str): The description text to display below the heading.
    """
    st.markdown(f"""
        <div style='text-align: center;'>
            <h3>{heading}</h3>
            <p>{description}</p>
        </div>
    """, unsafe_allow_html=True)


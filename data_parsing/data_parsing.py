from datasets import load_dataset
import requests
from bs4 import BeautifulSoup, PageElement, Tag
import re
from typing import List, Union
import os
from pathlib import Path
from tqdm import tqdm
import spacy


def sentenization(text):
    """Sentence segmentation using Spacy

    Args:
        text (str): Document text

    Returns:
        List[str]: A list of sentences
    """
    nlp = spacy.load("nb_core_news_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def getHTMLDocument(url: str) -> PageElement:
    """Get the HTML document body from a given URL

    Args:
        url (str): URL to fetch the document from

    Returns:
        PageElement: PageElement contains the HTML for document body
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    body = soup.find("div", {"id": "documentBody"})
    if body: 
        return body 
    else: 
        return None


def parse_all_tags(body: PageElement, do_sentenization: bool) -> List[str]:
    """Parse document body as a whole sequence

    Args:
        body (PageElement): body contains all elements
        do_sentenization(bool): if True, do sentence tokenization on the output document
    Returns:
        List[str]: a list of string size 1
    """
    all_text = body.get_text(separator=" ")
    all_text = re.sub(r"\s+", " ", all_text).strip()
    if do_sentenization:
        return sentenization(all_text)
    else: return [all_text]


def parse_paragraph_tag(body: PageElement) -> List[str]:
    """Parse document body and return list of paragraph items

    Args:
        body (PageElement): body contains all elements

    Returns:
        List[str]: a list of paragraphs
    """
    all_tags = body.find_all(["p"])
    results = [tag.get_text() for tag in all_tags]
    return results


def parse_avsnitt_tag(body: PageElement) -> List[str]:
    """Parse document body where there are AVSNITT tags and 
        return list of avsnitt strings

    Args:
        body (PageElement): body contans all elements

    Returns:
        List[str]: a list of avsnitt strings
    """
    # Find all table with data-level="2"
    results = []
    avsnitt_tables = body.find_all(["table"])
    for table in avsnitt_tables:
        sub_table_tags = table.find_all(["td"])
        for tag in sub_table_tags:
            # Remove avsnitt number -> not important to the text
            if not (tag.name == "td" and "avsnittNummer" in tag.get("class", [])):
                results.append(tag.get_text())
    return results
    

def is_inside_kapittel(tag: Tag) -> bool:
    """Check if a tag (usually the <p> tags) is inside a kapittel:
        <p> This is introduction text, before the kapittel </p>
        <div data-level="1">
            <p> This is kapittel text. </p>
            
    Args:
        tag (Tag): Tag to check

    Returns:
        bool: True if the tag has <div> parent tags, false otherwise
    """
    for parent in tag.parents:
        if parent.name == "div" and parent.get("data-level") == "1":
            return True
    return False


def get_kapittel_text(body: PageElement) -> List[str]:
    """Get content of kapittel and return a list of kapittel texts

    Args:
        body (PageElement): body contans all elements

    Returns:
        List[str]: a list of kapittel texts
    """
    kapittel_all_tags = body.find_all(["h2", "p", "blockquote"], recursive=False)
    kapittel_text = []
    for tag in kapittel_all_tags:
        if not(tag.name == "span" and tag.get("aria-label") == "Avidentifisert innhold"):
            text = tag.get_text(separator=" ")
            text = re.sub(r"\s+", " ", text).strip()
            # Only include non-empty text
            if text != "":
                kapittel_text.append(text)
    return kapittel_text


def parse_kapittel_tag(body: PageElement) -> dict:
    """Parse complicate document with Intrduction section and nested kapittel texts

    Args:
        body (PageElement): body contans all elements

    Returns:
        dict: dictionary contains introduction, kapittel and sub-kapittel texts
    """
    result_dict = {}
    # Find <p>, <td> elements
    kapittel_tags = body.find_all(["p", "td"])
    # If the tag is not inside the kapittel -> introduction text
    filtered_tags = [tag for tag in kapittel_tags if not is_inside_kapittel(tag)]
    intro_text = []
    for tag in filtered_tags:
        intro_text.append(tag.get_text())
    result_dict["KAPITTEL_0"] = {"KAPITTEL_0-0": intro_text}

    # Find all div elements with data-level="1"
    kapittel_divs = body.find_all("div", {"data-level": "1"})
    for kapittel_div in kapittel_divs:
        kapittel_id = kapittel_div["data-id"]
        sub_dict = {}
        # Find all div elements with data-level="2" within this kapittel div
        sub_kapittel_divs = kapittel_div.find_all('div', {'data-level': '2'})
        if len(sub_kapittel_divs) == 0: # KAPITTEL_x has no sub-kapittel
            kapittel_text = get_kapittel_text(kapittel_div)
            sub_dict[f"{kapittel_id}-0"] = kapittel_text
            result_dict[kapittel_id] = sub_dict
        else:
            # Get content of KAPITTEL_x
            kapittel_text = get_kapittel_text(kapittel_div)
            sub_dict[f"{kapittel_id}-0"] = kapittel_text
            result_dict[kapittel_id] = sub_dict
            # Get content of sub-KAPITTEL_x-y
            for sub_kapittel_div in sub_kapittel_divs:
                sub_kapittel_text = []
                # Get the sub-kapittel ID (e.g., KAPITTEL_1-1, KAPITTEL_1-2...)
                sub_kapittel_id = sub_kapittel_div["data-id"]
                sub_kapittel_tags = sub_kapittel_div.find_all(recursive=False)
                for tag in sub_kapittel_tags:
                    text = tag.get_text(separator=" ")
                    text = re.sub(r"\s+", " ", text).strip()
                    # Only include non-empty text
                    if text != "":
                        sub_kapittel_text.append(text)
                sub_dict[sub_kapittel_id] = sub_kapittel_text
            result_dict[kapittel_id] = sub_dict
    return result_dict


def parse_general_documents(url: str, do_segmentation: bool, do_sentenization: bool = False) -> Union[List[str], dict]:
    """Parse document from the given URL.

    Args:
        url (str): URL to fetch the document from.
        do_segmentation(bool): if True, do segmentation using keyword list, otherwise parse document as a long sequence.
        do_sentenization(bool): if True, do sentence tokenization. Only applicable when do_segmentation=False.

    Returns:
        Union[List[str], dict]: A list of strings containing the parsed document texts, 
        or a dictionary if this is a complicated nested-kapittel document.
    """
        
    keywords = {"KAPITTEL_", "AVSNITT_"}
    found_items = set()
    body = getHTMLDocument(url)
    if do_segmentation:
        for keyword in keywords:
            for tag in body.descendants:
                if getattr(tag, "name", None) is not None:
                    # Check if keyword is in the data-id attribute
                    if keyword in tag.get("data-id", ""):
                        # print(f"Keyword {keyword} is in the data-id attribute")
                        found_items.add(keyword)
        # print(found_items)
        if found_items:
            if "KAPITTEL_" in found_items:
                # print(f"Parse doc using KAPITTEL_")
                return parse_kapittel_tag(body)
            if "AVSNITT_" in found_items:
                # print(f"Parse doc using AVSNITT_")
                return parse_avsnitt_tag(body)
        else: 
            # print("Parse doc using paragraph")
            return parse_paragraph_tag(body)
    else:
        # If do_sentenization=True, then parse the sequence into a list of sentences
        if do_sentenization:
            return parse_all_tags(body = body, do_sentenization = do_sentenization)
        # Otherwise returns a long sequence
        else:
            return parse_all_tags(body = body, do_sentenization = do_sentenization)
    

def load_data(file_name: str) -> Union[List[str], dict]:
    """(TEST ONLY) Load json data and parse them

    Args:
        file_name (str): json input file

    Returns:
        Union[List[str], dict]: return parsed data in dict or list
    """
    except_dict = []
    current_path = Path(__file__).parent
    file_path = current_path.parent / file_name
    dataset_dict = load_dataset("json", data_files={"train":str(file_path)})
    for idx in tqdm(range(len(dataset_dict["train"]))):
        item = dataset_dict["train"][idx]
        try:
            url_ = f'https://lovdata.no{item["reference"]}'
            # print("url_ ", url_)
            parsed_doc = parse_general_documents(url=url_, do_segmentation=True, do_sentenization=True)  
            # print("parsed_doc:  ", parsed_doc)  
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except AttributeError as ae:
            print(f"AttributeError: {ae}")
            except_dict.append(f"{url_} | {ae}")
            print(f"Document: {idx} | URL: {url_}")
            print("Error parsing ", ae)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    print(except_dict)
    print("Total unparsed documents: ", len(except_dict))
    
    
    

    

# file_name = "data/lovdataValidation.json"
file_name = "data/sample3.json"
load_data(file_name=file_name)

"""
Interesting cases:
Avsnitt: 
    https://lovdata.no/dokument/HRSTR/avgjorelse/hr-2022-1401-u
Paragraph: 
    https://lovdata.no/dokument/LGSTR/avgjorelse/lg-2021-59642
Kapittel with p tag introduction: 
    https://lovdata.no/dokument/TRSTR/avgjorelse/trog-2020-186269
Kapittel with p, strong
    https://lovdata.no/dokument/LBSIV/avgjorelse/lb-2024-56875
Multi Kapittel: 
    https://lovdata.no/dokument/LGSIV/avgjorelse/lg-2022-66596
Multi-level kapittel
    https://lovdata.no/dokument/LBSTR/avgjorelse/lb-2023-183621
Table of text
    https://lovdata.no/dokument/LBSTR/avgjorelse/lb-2023-99549
Table of numbers, Kapittel, and p
    https://lovdata.no/dokument/TRSIV/avgjorelse/tosl-2024-55941
p tag with span and strong
    https://lovdata.no/dokument/LASTR/avgjorelse/la-2024-62221
p tag with span
    https://lovdata.no/dokument/LGSIV/avgjorelse/lg-2024-60413
"""

"""Exception cases
A lot of tables with numbers
    https://lovdata.no/dokument/LBSTR/avgjorelse/lb-2018-102939-2
"""

# url_ = "https://lovdata.no/dokument/LGSIV/avgjorelse/lg-2024-60413"
# url_doc = getHTMLDocument(url_)
# print(parse_kapittel_tag(url_doc))
# print(parse_general_documents(url=url_, do_segmentation=False))



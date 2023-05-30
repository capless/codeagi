from urllib.parse import urlparse

import tiktoken


def chunk_text(text, max_size):
    chunked_text = []
    current_chunk = ""
    for word in text.split():
        current_chunk += word + " "
        if len(current_chunk) > max_size:
            chunked_text.append(current_chunk)
            current_chunk = ""
    if current_chunk != "":
        chunked_text.append(current_chunk)
    return chunked_text


def chunk_tag_list(links_list, max_size):
    final_links_list = []
    link_href_list = []
    for link in links_list:
        if not isinstance(link, str):
            try:
                if link.attrs['href'] not in link_href_list:
                    link_href_list.append(link.attrs['href'])
                    final_links_list.append(str(link))
            except KeyError:
                pass
    enc = tiktoken.get_encoding("cl100k_base")
    result_list = []
    tag_list = ""
    for link in final_links_list:
        link_str = f"{link},"
        if len(enc.encode(tag_list + link_str)) < max_size:
            tag_list += link_str
        else:
            result_list.append(tag_list)
            tag_list = link_str

    result_list.append(tag_list)
    return result_list


def get_fqdn_list(url: str, link_list: list):
    up = urlparse(url)
    hostname = f"{up.scheme}://{up.hostname}"
    fqdn_list = []
    for i in link_list:
        link = i['link']
        if link.startswith('mailto:'):
            continue
        hn = urlparse(link).hostname
        if not hn:
            fqdn_list.append(f"{hostname}{link}")
        else:
            fqdn_list.append(link)
    return fqdn_list

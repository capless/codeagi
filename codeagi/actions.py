"""
Action classes for CodeAGI
"""

import json

import openai
import requests
import tiktoken
from bs4 import BeautifulSoup
from googlesearch import search

from codeagi.utils import chunk_tag_list, get_fqdn_list, chunk_text


class Action:
    """
    Base class for actions
    """

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class ResearchAction(Action):
    """
    Research action
    """
    search_prompt_template = """
        Given the following task: "{task}", create the best search 
        term to use for research using the following 
        JSON template {{"search_term": ?}}:
        """
    search_result_prompt_template = """
        Which link out of the list below is the best to use for research for {search_term}? 
        Generate a JSON response with the following template: {{"link": "result_list"}}
        {result_list}
        """
    crawl_prompt_template = """
        Given the following list of links return the URLs that will help the most while 
        researching to achieve the goal to "{goal}". Output the URLs in valid JSON using the 
        following template for your output [{{"link": "http://example.com"}}]: {links}
        """
    summary_prompt_template = """
        Given the body of the page, return the summary (with minimal text) and relevant code snippets from the page below for the goal of "{goal}" using the following JSON template to output valid JSON only:
        {{ "summary": "", "relevant_code_snippets": [] }}
        {body}
        """

    def __call__(self, task, goal):
        """
        Search Google for research
        Args:
            search_term: The search term to use
        """

        # Ask OpenAI what is the best link to use from the result list
        return self.search(task, goal)

    def read_page(self, url):
        """
        Read the page at the given URL
        Args:
            url: The URL to read
        Returns:
            The page content as a string
        """
        return BeautifulSoup(requests.get(url).content, 'html.parser')

    def call_gpt(self, prompt):
        """
        Call OpenAI with the given prompt
        Args:
            prompt: The prompt to use
        Returns:
            The response from OpenAI
        """
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_token_count = len(enc.encode(prompt))
        max_allowed_tokens = 3500 - prompt_token_count
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_allowed_tokens,
                n=1,
                stop=None,
                temperature=0.5,
            )

            return json.loads(response.choices[0].text)
        except (ConnectionResetError, requests.ConnectionError, json.JSONDecodeError):
            # Try again
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_allowed_tokens,
                n=1,
                stop=None,
                temperature=0.5,
            )
            print('Raw Response: ', response.choices[0].text)
            return json.loads(response.choices[0].text)

    def search(self, task, goal):
        """
        Ask OpenAI what is the best link to use from the result list
        Args:
            search_term: The search term to use
            result_list: The list of links to choose from
        Returns:
            The best link to use
        """
        search_term = self.call_gpt(self.search_prompt_template.format(task=task))['search_term']
        print('Search Term: ', search_term)
        result_list = list(search(search_term, num_results=10))
        print('Result List: ', result_list)
        prompt = self.search_result_prompt_template.format(search_term=search_term,
                                                           result_list=result_list)
        print('Prompt: ', prompt)
        best_link = self.call_gpt(prompt)['link']
        print('Best Link: ', best_link)
        page = self.read_page(best_link)
        links = chunk_tag_list(page.find_all('a'), 800)
        best_links = []
        no_requests = 0
        for chunk in links:
            no_requests += 1
            print(f'#{no_requests} OpenAI Request')
            crawl_prompt = self.crawl_prompt_template.format(goal=goal, links=chunk)
            best_links.extend(self.call_gpt(crawl_prompt))
        crawl_list = get_fqdn_list(best_link, best_links)
        # Loop through the crawl list and use OpenAI to summarize the page and return relevant code.
        for crawl_url in crawl_list:
            page = self.read_page(crawl_url)
            body_list = chunk_text(page.find('body').text, 800)
            for body in body_list:
                summary_prompt = self.summary_prompt_template.format(goal=goal, body=body)
                print('----------------------------------------------------------------------------------')
                print('Summary Prompt: ', summary_prompt)
                summary = self.call_gpt(summary_prompt)
                print('Summary: ', summary)
                print('----------------------------------------------------------------------------------')
                # Save the summary and relevant code snippets to Pinecone


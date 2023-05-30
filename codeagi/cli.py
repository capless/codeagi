import json

import click
import openai
import tiktoken
from envs import env

# Replace 'your_api_key' with your actual API key
openai.api_key = env('OPENAI_API_KEY')


PROMPT_TEMPLATE = (
    "Given the prompt to build something with code: '{build_prompt}', "
    "create a plan and a task list using GPT-3 in JSON format. Use the "
    "following JSON template: {{\"research_list\": [{{\"title\": \"Research Django\", "
    "\"action_type\": \"google_research\", \"search_term\": \"Django documentation\"}}, "
    "{{\"title\": \"Read Django documentation\", \"action_type\": \"read_documentation\", \"url\": \"https://docs.djangoproject.com/\"}}, "
    "{{\"title\": \"Clone a repository\", \"action_type\": \"clone_repo\", \"repo_url\": \"https://github.com/example/repo.git\"}}, "
    "{{\"title\": \"Check installed software\", \"action_type\": \"check_installed_software\", \"software\": \"Python\"}}, "
    "{{\"title\": \"Install dependencies\", \"action_type\": \"install_dependencies\", \"dependencies\": [\"Django\"]}}, "
    "{{\"title\": \"Write code\", \"action_type\": \"write_code\", \"code_snippet\": \"def example_function(): pass\"}}, "
    "{{\"title\": \"Execute a command\", \"action_type\": \"execute_command\", \"command\": \"python manage.py runserver\"}}, "
    "{{\"title\": \"Configure environment\", \"action_type\": \"configure_environment\", \"settings\": {{\"DEBUG\": \"True\"}}}}, "
    "{{\"title\": \"Run tests\", \"action_type\": \"run_tests\", \"test_command\": \"pytest\"}}, "
    "{{\"title\": \"Debug code\", \"action_type\": \"debug_code\", \"debugger\": \"pdb\"}}, "
    "{{\"title\": \"Refactor code\", \"action_type\": \"refactor_code\", \"refactoring_goal\": \"Improve readability\"}}, "
    "{{\"title\": \"Review code\", \"action_type\": \"review_code\", \"review_criteria\": [\"performance\", \"style\"]}}, "
    "{{\"title\": \"Create documentation\", \"action_type\": \"create_documentation\", \"documentation_type\": \"README\"}}, "
    "{{\"title\": \"Deploy application\", \"action_type\": \"deploy_application\", \"deployment_target\": \"Heroku\"}}, "
    "{{\"title\": \"Monitor application\", \"action_type\": \"monitor_application\", \"monitoring_tool\": \"New Relic\"}}, "
    "{{\"title\": \"Update application\", \"action_type\": \"update_application\", \"update_goal\": \"Add new feature\"}}], "
    "\"task_list\": [{{\"title\": \"Create Poetry Project\", \"action_type\": \"execute_command\", \"command\": \"poetry init\"}}]}}:"
)


@click.command()
@click.option('--prompt', prompt='Please provide a prompt for your application (web or mobile)',
              help='The prompt to generate a research task list.')
def main(prompt):
    click.echo('Generating research task list...')
    research_task_list = generate_research_task_list(prompt)

    while True:
        click.echo('\nResearch task list:')
        for task in research_task_list.get('research_list'):
            click.echo(f'- {task.get("title")}')
        click.echo('\nTask list:')
        for task in research_task_list.get('task_list'):
            click.echo(f'- {task.get("title")}')

        is_approved = click.confirm('\nDo you approve this research task list?')
        if is_approved:
            break
        else:
            if click.confirm('Would you like to provide suggestions to improve the research task list?'):
                suggestions = click.prompt('Please provide your suggestions')
                research_task_list = generate_research_task_list(prompt, suggestions)
            else:
                click.echo('No suggestions provided. Generating a new research task list...')
                research_task_list = generate_research_task_list(prompt)


def generate_research_task_list(prompt, suggestions=None):
    enc = tiktoken.get_encoding("cl100k_base")
    prompt_token_count = len(enc.encode(prompt))
    max_allowed_tokens = 3500 - prompt_token_count
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=PROMPT_TEMPLATE.format(build_prompt=prompt),
        temperature=0.5,
        max_tokens=max_allowed_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return json.loads(response.choices[0].text.strip())


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 23:25:59 2024

@author: Nathan
"""

import re
import os

os.chdir('C:/Users/natha/Downloads/MSc Thesis - Nathan/mainmatter')

def extract_context(input_file, output_file, search_string, context_length=150):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    search_string_escaped = re.escape(search_string)

    matches = re.finditer(f'.{{0,{context_length}}}{search_string_escaped}.{{0,{context_length}}}', content)

    with open(output_file, 'w', encoding='utf-8') as output:
        for match in matches:
            output.write(match.group() + '\n\n')  


input_file = '2 - M1.tex'   
output_file = 'output.txt'  
search_string = r'\('     

extract_context(input_file, output_file, search_string)

# %%

import re

def extract_between_strings(input_file, output_file, start_string, end_string):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.readlines() 

    # Escape the strings to avoid regex issueswith \( 
    start_string_escaped = re.escape(start_string)
    end_string_escaped = re.escape(end_string)

    pattern = f'{start_string_escaped}(.*?){end_string_escaped}'

    with open(output_file, 'w', encoding='utf-8') as output:
        for line in content:
            matches = re.findall(pattern, line)
            for match in matches:
                output.write(start_string + match.strip() + end_string + '&   \\\ \n')

input_file = 'output.txt'   
output_file = 'symbols_output.txt'  
start_string = r'\('      
end_string = r'\)'       

extract_between_strings(input_file, output_file, start_string, end_string)



# %% 
def remove_duplicates(input_file, output_file):
    unique_lines = set()  

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    with open(output_file, 'w', encoding='utf-8') as output:
        for line in lines:
            if line not in unique_lines:
                output.write(line)  
                unique_lines.add(line)  

input_file = 'symbols_output.txt'   
output_file = 'symbols_output_for_latex.txt'  

remove_duplicates(input_file, output_file)

# %%

def capitalize_first_after_ampersand(text):
    if '&' in text:
        before_ampersand, after_ampersand = text.split('&', 1)
        after_ampersand_cleaned = after_ampersand.lstrip()
        if after_ampersand_cleaned:
            after_ampersand_cleaned = after_ampersand_cleaned[0].upper() + after_ampersand_cleaned[1:]
        return before_ampersand + '&' + after_ampersand_cleaned
    else:
        return text

def sort_and_capitalize_file(input_file, output_file):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    processed_lines = [capitalize_first_after_ampersand(line.strip()) for line in lines]

    sorted_lines = sorted(processed_lines, key=lambda line: line.split('&', 1)[1].replace(' ', '') if '&' in line else '')

    with open(output_file, 'w') as outfile:
        for line in sorted_lines:
            outfile.write(line + '\n')

            
input_filename = 'symbols_output_for_latex.txt'
output_filename = 'alphabetically_sorted.txt'
sort_and_capitalize_file(input_filename, output_filename)

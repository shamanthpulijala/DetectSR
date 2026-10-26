import re

file_path = r'E:\code things\DL_PROJECT\assets\index-w9f63tud.js'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Remove Bulk URL from Ie enum
if ',l.BULK_URL="Bulk URL"' in content:
    content = content.replace(',l.BULK_URL="Bulk URL"', '')
    print('Removed Bulk URL enum (version A)')
elif 'l.BULK_URL="Bulk URL",' in content:
    content = content.replace('l.BULK_URL="Bulk URL",', '')
    print('Removed Bulk URL enum (version B)')
elif 'l.BULK_URL="Bulk URL"' in content:
    print("Found exact string")
    content = content.replace('l.BULK_URL="Bulk URL"', '')

# 2. Fix the onClick tab glitch
old_onclick = 'onClick:()=>z(B)'
new_onclick = 'onClick:()=>{z(B);M()}'
if old_onclick in content:
    content = content.replace(old_onclick, new_onclick)
    print('Fixed onClick glitch')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Frontend patch successful")

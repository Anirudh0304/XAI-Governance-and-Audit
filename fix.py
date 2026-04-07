content = open("src/dashboard.py").read()

content = content.replace("width='stretch'", "use_container_width=True")
content = content.replace("width='content'", "use_container_width=False")

open("src/dashboard.py", "w").write(content)

print("Fixed")
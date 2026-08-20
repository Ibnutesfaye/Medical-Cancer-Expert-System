with open('main.py', encoding='utf-8') as f:
    content = f.read()

register_block = (
    '\n@app.post("/auth/register", status_code=201)\n'
    'async def register(request: LoginRequest):\n'
    '    """Register a new user account."""\n'
    '    return auth_service.register_user(username=request.username, password=request.password)\n'
    '\n\n'
)

target = '@app.post("/auth/logout")'
if target in content:
    content = content.replace(target, register_block + target, 1)
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('register endpoint added OK')
else:
    print('ERROR: target not found')

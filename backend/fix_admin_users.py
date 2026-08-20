with open('main.py', encoding='utf-8') as f:
    content = f.read()

old = (
    '@app.get("/admin/users")\n'
    'async def list_users(token_data: TokenData = Depends(auth_service.get_current_user)):\n'
    '    """List all users (in-memory - only shows current user)."""\n'
    '    if not token_data.is_admin:\n'
    '        raise HTTPException(status_code=403, detail="Admin access required")\n'
    '    return [{\n'
    '        "id": 1,\n'
    '        "username": token_data.username,\n'
    '        "email": None,\n'
    '        "full_name": "Admin User",\n'
    '        "is_admin": token_data.is_admin,\n'
    '        "is_active": True,\n'
    '        "created_at": "2024-01-01T00:00:00",\n'
    '        "updated_at": "2024-01-01T00:00:00"\n'
    '    }]'
)

new = (
    '@app.get("/admin/users")\n'
    'async def list_users(token_data: TokenData = Depends(auth_service.get_current_user)):\n'
    '    """List all registered users."""\n'
    '    if not token_data.is_admin:\n'
    '        raise HTTPException(status_code=403, detail="Admin access required")\n'
    '    return auth_service.list_users()'
)

if old in content:
    content = content.replace(old, new, 1)
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('admin/users fixed OK')
else:
    print('ERROR: target not found - checking...')
    if '@app.get("/admin/users")' in content:
        print('endpoint exists but text differs')
    else:
        print('endpoint missing entirely')

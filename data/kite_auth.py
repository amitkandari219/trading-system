"""
Kite Connect authentication helper.

Kite requires a daily login flow:
  1. Generate login URL (user opens in browser)
  2. User logs in → redirected with request_token in URL
  3. Exchange request_token for access_token (valid for 1 day)

Usage:
    # First time / daily login:
    python -m data.kite_auth --login

    # After login, access token is saved to .env.kite
    # All other scripts read from there automatically.

    # Check if token is still valid:
    python -m data.kite_auth --check
"""

import os
import sys
import json
from datetime import datetime

TOKEN_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env.kite')


def get_credentials():
    """Read API key and secret from environment or .env.kite."""
    # Try environment first
    api_key = os.environ.get('KITE_API_KEY')
    api_secret = os.environ.get('KITE_API_SECRET')

    # Fall back to .env.kite
    if not api_key and os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            for line in f:
                line = line.strip()
                if line.startswith('KITE_API_KEY='):
                    api_key = line.split('=', 1)[1].strip()
                elif line.startswith('KITE_API_SECRET='):
                    api_secret = line.split('=', 1)[1].strip()

    return api_key, api_secret


def get_access_token():
    """Read saved access token. Returns None if expired or missing."""
    if not os.path.exists(TOKEN_FILE):
        return None

    access_token = None
    token_date = None

    with open(TOKEN_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith('KITE_ACCESS_TOKEN='):
                access_token = line.split('=', 1)[1].strip()
            elif line.startswith('KITE_TOKEN_DATE='):
                token_date = line.split('=', 1)[1].strip()

    # Token is valid for 1 day only
    if token_date and token_date != datetime.now().strftime('%Y-%m-%d'):
        print(f"Access token expired (from {token_date}). Run: python -m data.kite_auth --login")
        return None

    return access_token


def get_kite():
    """Get an authenticated KiteConnect instance. Returns None if not authenticated."""
    from kiteconnect import KiteConnect

    api_key, _ = get_credentials()
    access_token = get_access_token()

    if not api_key or not access_token:
        return None

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def save_token(api_key, api_secret, access_token):
    """Save credentials and access token to .env.kite."""
    with open(TOKEN_FILE, 'w') as f:
        f.write(f"KITE_API_KEY={api_key}\n")
        f.write(f"KITE_API_SECRET={api_secret}\n")
        f.write(f"KITE_ACCESS_TOKEN={access_token}\n")
        f.write(f"KITE_TOKEN_DATE={datetime.now().strftime('%Y-%m-%d')}\n")
    os.chmod(TOKEN_FILE, 0o600)  # Read/write only for owner
    print(f"Saved to {TOKEN_FILE}")


def login_flow():
    """Interactive login flow."""
    from kiteconnect import KiteConnect

    api_key, api_secret = get_credentials()

    if not api_key:
        api_key = input("Enter Kite API Key: ").strip()
    if not api_secret:
        api_secret = input("Enter Kite API Secret: ").strip()

    if not api_key or not api_secret:
        print("ERROR: API key and secret are required.")
        print("Get them from https://developers.kite.trade/apps")
        sys.exit(1)

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print(f"\n1. Open this URL in your browser:")
    print(f"   {login_url}")
    print(f"\n2. Log in with your Zerodha credentials")
    print(f"3. After login, you'll be redirected to a URL like:")
    print(f"   https://127.0.0.1/?request_token=XXXX&action=login&status=success")
    print(f"\n4. Copy the request_token from the URL")

    request_token = input("\nPaste request_token here: ").strip()

    if not request_token:
        print("ERROR: No request token provided.")
        sys.exit(1)

    try:
        data = kite.generate_session(request_token, api_secret=api_secret)
        access_token = data['access_token']
        kite.set_access_token(access_token)

        # Verify
        profile = kite.profile()
        print(f"\nLogged in as: {profile['user_name']} ({profile['user_id']})")
        print(f"Email: {profile['email']}")

        save_token(api_key, api_secret, access_token)
        print("Authentication successful. Token valid until tomorrow morning.")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("Make sure the request_token is fresh (they expire in ~60 seconds).")
        sys.exit(1)


def check_token():
    """Check if current token is valid."""
    kite = get_kite()
    if not kite:
        print("No valid token. Run: python -m data.kite_auth --login")
        return False

    try:
        profile = kite.profile()
        print(f"Token valid. Logged in as: {profile['user_name']} ({profile['user_id']})")
        return True
    except Exception as e:
        print(f"Token invalid: {e}")
        print("Run: python -m data.kite_auth --login")
        return False


if __name__ == '__main__':
    if '--login' in sys.argv:
        login_flow()
    elif '--check' in sys.argv:
        check_token()
    else:
        print("Usage:")
        print("  python -m data.kite_auth --login   # Daily login flow")
        print("  python -m data.kite_auth --check   # Check token validity")

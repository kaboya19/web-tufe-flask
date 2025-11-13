#!/usr/bin/env python3
"""
VAPID Key Generator for Web Push Notifications

This script generates VAPID (Voluntary Application Server Identification) keys
for web push notifications. These keys are used to identify your application
server to push services.

Usage:
    python generate_vapid_keys.py [email]

If email is not provided, it will use a default email.
"""

import base64
import json
import sys
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


def generate_vapid_keys(email='mailto:webtufe@example.com'):
    """
    Generate VAPID keys for web push notifications.
    
    Args:
        email: Email address or mailto: URI to use in VAPID claims
        
    Returns:
        tuple: (public_key, private_key) both in base64 URL-safe format
    """
    # Generate a new EC key pair using P-256 curve (required for VAPID)
    private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
    public_key = private_key.public_key()
    
    # Get the public key in compressed format (X, Y coordinates)
    # VAPID uses uncompressed format (0x04 + X + Y)
    public_key_numbers = public_key.public_numbers()
    x_bytes = public_key_numbers.x.to_bytes(32, byteorder='big')
    y_bytes = public_key_numbers.y.to_bytes(32, byteorder='big')
    
    # Uncompressed public key: 0x04 prefix + X coordinate + Y coordinate
    uncompressed_public_key = b'\x04' + x_bytes + y_bytes
    
    # Encode public key as base64 URL-safe (without padding)
    public_key_b64 = base64.urlsafe_b64encode(uncompressed_public_key).decode('utf-8').rstrip('=')
    
    # Get private key in DER format (PKCS8)
    private_key_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Encode private key as base64 URL-safe (without padding)
    private_key_b64 = base64.urlsafe_b64encode(private_key_der).decode('utf-8').rstrip('=')
    
    # Also get private key in PEM format (for backup/storage)
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')
    
    return {
        'public_key': public_key_b64,
        'private_key': private_key_b64,
        'private_key_pem': private_key_pem,
        'email': email
    }


def format_for_env_file(keys):
    """
    Format keys for .env file.
    
    Args:
        keys: Dictionary containing public_key, private_key, and email
        
    Returns:
        str: Formatted string for .env file
    """
    return f"""# VAPID Keys for Web Push Notifications
# Generated email: {keys['email']}
VAPID_PUBLIC_KEY={keys['public_key']}
VAPID_PRIVATE_KEY={keys['private_key']}
VAPID_CLAIM_EMAIL={keys['email']}
"""


def format_for_json(keys):
    """
    Format keys as JSON.
    
    Args:
        keys: Dictionary containing public_key, private_key, and email
        
    Returns:
        str: JSON formatted string
    """
    # Don't include PEM in JSON output (too long)
    json_keys = {
        'public_key': keys['public_key'],
        'private_key': keys['private_key'],
        'email': keys['email']
    }
    return json.dumps(json_keys, indent=2)


def main():
    """Main function to generate and display VAPID keys."""
    # Get email from command line argument or use default
    email = sys.argv[1] if len(sys.argv) > 1 else 'mailto:webtufe@example.com'
    
    # Ensure email starts with 'mailto:' if it's an email address
    if '@' in email and not email.startswith('mailto:'):
        email = f'mailto:{email}'
    elif not email.startswith('mailto:') and not email.startswith('https://'):
        email = f'mailto:{email}'
    
    print("=" * 70)
    print("VAPID Key Generator for Web Push Notifications")
    print("=" * 70)
    print(f"\nGenerating VAPID keys for: {email}\n")
    
    try:
        # Generate keys
        keys = generate_vapid_keys(email)
        
        print("✅ VAPID keys generated successfully!\n")
        print("=" * 70)
        print("PUBLIC KEY (Base64 URL-safe):")
        print("=" * 70)
        print(keys['public_key'])
        print()
        
        print("=" * 70)
        print("PRIVATE KEY (Base64 URL-safe):")
        print("=" * 70)
        print(keys['private_key'])
        print()
        
        print("=" * 70)
        print("PRIVATE KEY (PEM format - for backup):")
        print("=" * 70)
        print(keys['private_key_pem'])
        print()
        
        print("=" * 70)
        print(".env FILE FORMAT:")
        print("=" * 70)
        print(format_for_env_file(keys))
        print()
        
        print("=" * 70)
        print("JSON FORMAT:")
        print("=" * 70)
        print(format_for_json(keys))
        print()
        
        print("=" * 70)
        print("📝 IMPORTANT NOTES:")
        print("=" * 70)
        print("1. Keep your PRIVATE KEY secure and never share it publicly!")
        print("2. Add these keys to your .env file or environment variables")
        print("3. Use the PUBLIC KEY in your frontend JavaScript code")
        print("4. The PRIVATE KEY is used only on the server side")
        print("5. VAPID_CLAIM_EMAIL should be a valid email or mailto: URI")
        print("6. Save the PEM format private key as a backup")
        print("=" * 70)
        
        # Ask if user wants to save to file
        save_to_file = input("\n💾 Save keys to vapid_keys.json? (y/n): ").strip().lower()
        if save_to_file == 'y':
            with open('vapid_keys.json', 'w') as f:
                json.dump(keys, f, indent=2)
            print("✅ Keys saved to vapid_keys.json")
            print("⚠️  Remember to keep this file secure and add it to .gitignore!")
        
        save_env = input("💾 Save to .env file? (y/n): ").strip().lower()
        if save_env == 'y':
            env_content = format_for_env_file(keys)
            with open('.env', 'a') as f:
                f.write('\n')
                f.write(env_content)
            print("✅ Keys appended to .env file")
            print("⚠️  Make sure .env is in your .gitignore file!")
        
    except Exception as e:
        print(f"❌ Error generating VAPID keys: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


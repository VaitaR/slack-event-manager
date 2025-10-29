#!/usr/bin/env python3
"""Telegram Authentication Script.

Creates session file for Telethon user client using phone + SMS or QR code login.

Usage:
    python scripts/telegram_qr_auth.py [phone] [code]

    If no arguments provided, will try QR code login.
    If phone provided, will try SMS verification first, then QR code if fails.

Requirements:
    - TELEGRAM_API_ID in .env
    - TELEGRAM_API_HASH in .env
    - Telegram app on your phone

Steps:
1. Run this script with phone number: python scripts/telegram_qr_auth.py
2. Enter verification code when prompted
3. If SMS fails, will fallback to QR code
4. Session file will be created automatically
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from telethon import TelegramClient

try:
    import qrcode

    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# Load environment variables
load_dotenv()


def print_qr_code(url: str) -> None:
    """Print QR code in terminal using ASCII art."""
    if QR_AVAILABLE:
        try:
            qr = qrcode.QRCode(version=1, box_size=2, border=1)
            qr.add_data(url)
            qr.make(fit=True)

            print("\n" + "=" * 60)
            print("QR CODE (ASCII ART):")
            print("=" * 60)
            qr.print_ascii(invert=True)
            print("=" * 60)
        except Exception as e:
            print(f"⚠️  Could not generate QR code: {e}")
            print("Using URL instead...")
    else:
        print("⚠️  qrcode library not available. Install with: pip install qrcode[pil]")
        print("Using URL instead...")


async def authenticate_with_qr() -> None:
    """Run QR code authentication flow using proper QR login method."""
    # Get credentials from environment
    api_id_str = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")

    if not api_id_str or not api_hash:
        print("❌ Error: TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env")
        print("\nTo get credentials:")
        print("1. Go to https://my.telegram.org")
        print("2. Log in with your phone number")
        print("3. Go to 'API development tools'")
        print("4. Create an application")
        print("5. Copy API ID and API hash to .env")
        sys.exit(1)

    try:
        api_id = int(api_id_str)
    except ValueError:
        print(f"❌ Error: TELEGRAM_API_ID must be an integer, got: {api_id_str}")
        sys.exit(1)

    # Session file path
    session_path = "data/telegram_session.session"
    Path("data").mkdir(exist_ok=True)

    print("=" * 60)
    print("Telegram QR Code Authentication")
    print("=" * 60)
    print(f"API ID: {api_id}")
    print(f"Session file: {session_path}")
    print()

    # Create client with file-based session (SQLite)
    client = TelegramClient(session_path, api_id, api_hash)

    try:
        print("🔐 Starting QR code authentication...")
        print()
        print("📱 Instructions:")
        print("1. Open Telegram app on your phone")
        print("2. Go to Settings > Devices > Link Desktop Device")
        print("3. Scan the QR code that will appear below")
        print("4. Confirm login in the app")
        print()

        # Connect to Telegram
        await client.connect()

        # Check if already authorized
        if await client.is_user_authorized():
            print("✅ Already authorized! Getting user info...")
        else:
            print("🔐 Starting QR code login process...")

            # Use native qr_login method
            try:
                qr_login = await client.qr_login()

                print("\n" + "=" * 60)
                print("SCAN THIS QR CODE WITH YOUR TELEGRAM APP:")
                print("=" * 60)
                print(qr_login.url)
                print("=" * 60)

                # Print QR code in terminal
                print_qr_code(qr_login.url)

                print("\n📱 Instructions:")
                print("1. Open Telegram app on your phone")
                print("2. Go to Settings > Devices > Link Desktop Device")
                print("3. Scan the QR code above (ASCII art or URL)")
                print("4. Confirm login in the app")
                print("=" * 60)

                # Wait for QR code to be scanned
                print("\n⏳ Waiting for QR code to be scanned...")
                print("(Press Ctrl+C to cancel)")
                print(
                    "\n💡 TIP: You can also copy the URL above and open it in your browser"
                )
                print(
                    "   or use any QR code generator to create a visual QR code from the URL"
                )

                # Wait for QR code to be scanned (native method)
                try:
                    await qr_login.wait()
                    print("\n✅ QR code scanned successfully!")
                except KeyboardInterrupt:
                    print("\n⚠️  QR login cancelled by user")
                    sys.exit(1)
                except Exception as e:
                    print(f"\n❌ QR login failed: {e}")
                    print("Please try again by running the script again")
                    sys.exit(1)

            except Exception as e:
                print(f"❌ Failed to generate QR code: {e}")
                print("Please check your API credentials and try again")
                sys.exit(1)

        # Get current user info
        me = await client.get_me()

        print()
        print("=" * 60)
        print("✅ Authentication successful!")
        print("=" * 60)
        print(f"User ID: {me.id}")
        print(f"Username: @{me.username}" if me.username else "Username: (not set)")
        print(f"First name: {me.first_name}")
        print(f"Last name: {me.last_name}" if me.last_name else "")
        print()

        # Session is automatically saved to file by Telethon
        print(f"Session saved to: {session_path}")
        print()
        print("You can now use Telegram integration in the pipeline.")
        print("Run: python scripts/run_multi_source_pipeline.py --source telegram")

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\nTroubleshooting:")
        print("- Make sure you have the latest version of Telegram app")
        print("- Try logging out and logging back in to Telegram app")
        print("- Check your internet connection")
        print("- Make sure API credentials are correct")
        sys.exit(1)

    finally:
        await client.disconnect()


def main() -> None:
    """Main entry point."""
    print()
    print("This script will authenticate your Telegram account using QR code.")
    print("You will need:")
    print("  - Telegram app on your phone")
    print("  - QR code scanner in Telegram app")
    print()

    try:
        asyncio.run(authenticate_with_qr())
    except KeyboardInterrupt:
        print("\n\n⚠️  Authentication cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

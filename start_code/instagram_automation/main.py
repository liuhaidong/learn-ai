"""Instagram Automation - Main Entry Point"""

import asyncio
import os
import sys
from datetime import time
from dotenv import load_dotenv

from config.settings import settings
from instagram.account_manager import TwoAccountManager
from services.content_generator import ContentGeneratorService
from services.interaction import InteractionService
from services.cost_monitor import CostMonitor
from services.rate_limiter import TwoAccountRateLimiter


async def main():
    """Main entry point for Instagram automation"""

    print("=" * 70)
    print("🚀 INSTAGRAM AUTOMATION - E-COMMERCE CROSS-BORDER SALES")
    print("=" * 70)
    print(f"📊 Budget: ${settings.monthly_budget}/month (${settings.daily_budget:.2f}/day)")
    print(f"👥 Accounts: 2 test accounts configured")
    print(f"📱 Product Categories: 6 categories (charging_cable, charger, earbuds, phone_film, phone_case, noise_cancelling_headphone)")
    print("=" * 70)

    # Load environment variables
    load_dotenv()

    # Check required configuration
    if not settings.openai_api_key:
        print("❌ ERROR: OPENAI_API_KEY not set in .env file")
        print("   Please add your OpenAI API key to continue")
        return

    if not settings.midjourney_api_key:
        print("❌ ERROR: MIDJOURNEY_API_KEY not set in .env file")
        print("   Please add your Midjourney API key to continue")
        return

    # Initialize components
    account_manager = TwoAccountManager()
    rate_limiter = TwoAccountRateLimiter()
    cost_monitor = CostMonitor()
    content_generator = ContentGeneratorService(account_manager)
    interaction_service = InteractionService(account_manager, rate_limiter)

    # Configure accounts
    account_configs = {
        1: {
            'username': settings.instagram_account_1,
            'password': settings.instagram_password_1,
            'primary_category': settings.account_1_primary_category,
            'secondary_categories': settings.account_1_secondary_categories.split(',') if settings.account_1_secondary_categories else [],
            'proxy': settings.proxy_url if settings.use_proxy else None
        },
        2: {
            'username': settings.instagram_account_2,
            'password': settings.instagram_password_2,
            'primary_category': settings.account_2_primary_category,
            'secondary_categories': settings.account_2_secondary_categories.split(',') if settings.account_2_secondary_categories else [],
            'proxy': settings.proxy_url if settings.use_proxy else None
        }
    }

    # Setup accounts
    print("\n⚙️  Setting up Instagram accounts...")
    await account_manager.setup_accounts(account_configs)

    active_accounts = account_manager.get_active_accounts()
    if not active_accounts:
        print("❌ ERROR: No active accounts found")
        print("   Please configure test accounts in .env file")
        print("   Format: INSTAGRAM_TEST_ACCOUNT_1=test_username_1")
        print("           INSTAGRAM_TEST_PASSWORD_1=your_password")
        return

    print(f"✅ Ready with {len(active_accounts)} active account(s)")

    try:
        # Main menu
        while True:
            print("\n" + "=" * 70)
            print("📋 MAIN MENU")
            print("=" * 70)
            print("1.  Generate Content (E-commerce focused)")
            print("2.  Run Daily Interactions")
            print("3.  Check Cost Report")
            print("4.  View Account Status")
            print("5.  Exit")
            print("=" * 70)

            choice = input("Enter your choice (1-5): ").strip()

            if choice == '1':
                await content_generation_menu(content_generator, cost_monitor)

            elif choice == '2':
                await run_interactions_menu(interaction_service, active_accounts)

            elif choice == '3':
                await show_cost_report(cost_monitor)

            elif choice == '4':
                await show_account_status(account_manager)

            elif choice == '5':
                print("\n👋 Logging out all accounts...")
                await account_manager.logout_all()
                print("✅ Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please try again.")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        await account_manager.logout_all()
        print("✅ All accounts logged out")


async def content_generation_menu(content_generator: ContentGeneratorService, cost_monitor: CostMonitor):
    """Content generation submenu"""

    print("\n" + "=" * 70)
    print("📝 CONTENT GENERATION")
    print("=" * 70)

    while True:
        print("\n1.  Generate Single Post")
        print("2.  Generate Batch Posts (3)")
        print("3.  Generate Flash Sale Post")
        print("4.  Back to Main Menu")
        print("=" * 70)

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            account_id = input("Enter account ID (1 or 2): ").strip()
            category = input("Enter category (charging_cable, charger, earbuds, phone_film, phone_case, noise_cancelling_headphone): ").strip()

            if account_id in ['1', '2']:
                await content_generator.generate_post_content(
                    account_id=int(account_id),
                    category=category
                )

        elif choice == '2':
            account_id = input("Enter account ID (1 or 2): ").strip()
            if account_id in ['1', '2']:
                await content_generator.generate_batch_posts(
                    account_id=int(account_id),
                    num_posts=3
                )

        elif choice == '3':
            account_id = input("Enter account ID (1 or 2): ").strip()
            category = input("Enter category: ").strip()
            discount = input("Enter discount % (10, 20, 30): ").strip()

            if account_id in ['1', '2'] and discount.isdigit():
                await content_generator.generate_value_deal_post(
                    account_id=int(account_id),
                    category=category,
                    discount_percent=int(discount)
                )

        elif choice == '4':
            break

        else:
            print("❌ Invalid choice")


async def run_interactions_menu(interaction_service: InteractionService, active_accounts):
    """Run interactions submenu"""

    print("\n" + "=" * 70)
    print("👥 INTERACTION AUTOMATION")
    print("=" * 70)

    if not active_accounts:
        print("❌ No active accounts available")
        return

    for account_id in active_accounts:
        print(f"\nRunning interactions for account {account_id}...")
        await interaction_service.daily_interaction_task(account_id)

    print("\n✅ All interactions complete!")


async def show_cost_report(cost_monitor: CostMonitor):
    """Show cost report"""

    print("\n" + "=" * 70)
    print("💰 COST REPORT")
    print("=" * 70)

    report = await cost_monitor.generate_daily_report()

    summary = await cost_monitor.get_cost_summary()

    print(f"\n📊 Cost Summary:")
    print(f"   Daily Cost: ${summary['current_day_cost']:.2f}")
    print(f"   Daily Budget: ${summary['daily_budget']:.2f}")
    print(f"   Daily Remaining: ${summary['daily_remaining']:.2f}")
    print(f"\n   Monthly Cost: ${summary['current_month_cost']:.2f}")
    print(f"   Monthly Budget: ${summary['monthly_budget']:.2f}")
    print(f"   Monthly Remaining: ${summary['monthly_remaining']:.2f}")
    print(f"   Budget Used: {summary['budget_percentage']:.1f}%")


async def show_account_status(account_manager: TwoAccountManager):
    """Show account status"""

    print("\n" + "=" * 70)
    print("👤 ACCOUNT STATUS")
    print("=" * 70)

    active_accounts = account_manager.get_active_accounts()
    usage_stats = account_manager.get_usage_stats()

    print(f"\nActive Accounts: {len(active_accounts)}")
    for account_id in active_accounts:
        print(f"\nAccount {account_id}:")
        print(f"   Status: Active")
        print(f"   Actions Today: {usage_stats.get(account_id, 0)}")


if __name__ == "__main__":
    asyncio.run(main())

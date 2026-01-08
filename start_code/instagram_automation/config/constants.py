"""Instagram Automation - Constants for 6 Product Categories"""

from typing import Dict, List, Any

# 6 Product Categories for E-commerce Cross-border Sales
PRODUCT_CATEGORIES = {
    "charging_cable": {
        "name": "Charging Cable",
        "hashtags": {
            "primary": [
                "#datacable", "#charger", "#usbcable", "#typec", "#chargingcable",
                "#usb", "#fastcharging", "#technology", "#gadgets", "#electronics"
            ],
            "niche": [
                "#fastcharging", "#usbcables", "#lightningcable", "#iphonecharger",
                "#durablecable", "#typeccable", "#usbcharger", "#charginggear"
            ],
            "long_tail": [
                "#cablequality", "#durablecable", "#fastchargingcable",
                "#chargingaccessories", "#phonecable", "#datacables",
                "#fastcharger", "#techgadgets", "#mobileaccessories"
            ]
        },
        "content_themes": [
            "durability_test", "fast_charging_review", "cable_comparison",
            "brand_reliability", "value_for_money"
        ],
        "price_range": {"min": 5, "max": 30},
        "key_features": ["fast_charging", "durable_material", "tangle_free", "compatible_devices"],
        "target_audience": "tech-savvy users looking for reliable charging solutions"
    },
    "charger": {
        "name": "Charger",
        "hashtags": {
            "primary": [
                "#charger", "#fastcharging", "#powerbank", "#usb",
                "#charging", "#technology", "#gadgets", "#electronics", "#tech"
            ],
            "niche": [
                "#fastcharger", "#usbcharger", "#carcharger", "#quickcharge",
                "#chargingstation", "#wirelesscharger", "#usbcharging", "#mobilecharger"
            ],
            "long_tail": [
                "#chargingstation", "#techgadgets", "#mobileaccessories",
                "#fastcharge", "#chargingcables", "#poweradapter", "#usbcharging"
            ]
        },
        "content_themes": [
            "speed_test", "portability_review", "safety_features",
            "brand_comparison", "value_deal"
        ],
        "price_range": {"min": 10, "max": 50},
        "key_features": ["quick_charge", "compact_design", "safety_protection", "universal_compatibility"],
        "target_audience": "mobile users needing reliable charging solutions"
    },
    "earbuds": {
        "name": "Earbuds",
        "hashtags": {
            "primary": [
                "#earbuds", "#headphones", "#bluetooth", "#music", "#wireless",
                "#audio", "#gadgets", "#technology", "#tech"
            ],
            "niche": [
                "#airpods", "#wirelessearbuds", "#bluetoothearbuds",
                "#wirelessheadphones", "#earbudswireless", "#wirelessaudio",
                "#bluetoothheadphones", "#earbud"
            ],
            "long_tail": [
                "#audiophile", "#musiclover", "#earbudslover",
                "#wirelessaudio", "#earbudsbluetooth", "#headphonegaming",
                "#earphone", "#earphones", "#headset"
            ]
        },
        "content_themes": [
            "sound_quality_test", "noise_reduction", "battery_life",
            "comfort_review", "value_comparison"
        ],
        "price_range": {"min": 20, "max": 100},
        "key_features": ["noise_cancellation", "long_battery", "comfortable_fit", "premium_sound"],
        "target_audience": "music enthusiasts, commuters, and gamers"
    },
    "phone_film": {
        "name": "Screen Protector",
        "hashtags": {
            "primary": [
                "#phonecase", "#screenprotector", "#temperedglass",
                "#phoneaccessories", "#gadgets", "#technology", "#tech"
            ],
            "niche": [
                "#screenprotector", "#iphoneprotector", "#temperedglass",
                "#phoneprotection", "#screenprotection", "#iphonecase",
                "#temperedglassscreen", "#screenfilm"
            ],
            "long_tail": [
                "#screenprotection", "#durableprotector", "#clearprotector",
                "#protectorfilm", "#phonescreen", "#glassprotector",
                "#screenprotectors"
            ]
        },
        "content_themes": [
            "installation_demo", "scratch_resistance_test", "clarity_review",
            "durability_test", "value_deal"
        ],
        "price_range": {"min": 5, "max": 20},
        "key_features": ["scratch_resistant", "easy_install", "high_clarity", "bubble_free"],
        "target_audience": "smartphone owners wanting to protect their devices"
    },
    "phone_case": {
        "name": "Phone Case",
        "hashtags": {
            "primary": [
                "#phonecase", "#caseiphone", "#phoneaccessories",
                "#gadgets", "#technology", "#tech", "#electronics"
            ],
            "niche": [
                "#iphonecase", "#phonecovers", "#customcase", "#designercase",
                "#phonecasefashion", "#protectivecase", "#slimcase", "#ruggedcase"
            ],
            "long_tail": [
                "#phonecasefashion", "#protectivecase", "#slimcase",
                "#ruggedcase", "#caseiphone", "#phonecovers", "#phoneaccessories"
            ]
        },
        "content_themes": [
            "design_showcase", "protection_test", "material_quality",
            "color_variations", "value_deal"
        ],
        "price_range": {"min": 10, "max": 40},
        "key_features": ["shock_absorption", "slim_design", "premium_material", "wireless_charging_compatible"],
        "target_audience": "fashion-conscious smartphone users wanting protection"
    },
    "noise_cancelling_headphone": {
        "name": "Noise-Cancelling Headphones",
        "hashtags": {
            "primary": [
                "#headphones", "#noise", "#audiophile", "#bluetooth",
                "#audio", "#gadgets", "#technology", "#tech"
            ],
            "niche": [
                "#noisecancelling", "#wirelessheadphones", "#headphonebluetooth",
                "#premiumaudio", "#noisecancelling", "#wirelessheadphone",
                "#headphonesreview", "#bluetoothheadphones"
            ],
            "long_tail": [
                "#noisecancelling", "#headphonegaming", "#audiophilegear",
                "#wirelessheadphone", "#headphone", "#headset", "#headphonesbluetooth"
            ]
        },
        "content_themes": [
            "noise_cancellation_test", "sound_isolation", "comfort_long_term",
            "brand_comparison", "value_deal"
        ],
        "price_range": {"min": 50, "max": 300},
        "key_features": ["active_noise_cancellation", "high_resolution_audio", "long_battery", "comfortable_design"],
        "target_audience": "commuters, travelers, and audiophiles"
    }
}

# Content Type Distribution for E-commerce
CONTENT_TYPE_DISTRIBUTION = {
    "product_review": 0.35,      # Product review and showcase
    "value_deal": 0.25,          # Special offers and deals
    "feature_highlight": 0.20,     # Key feature showcase
    "comparison": 0.10,            # Product comparison
    "customer_testimonial": 0.05,   # User reviews/feedback
    "installation_guide": 0.05         # How-to/use guide
}

# E-commerce Specific Hashtags
ECOMMERCE_HASHTAGS = [
    "#shopnow", "#limitedoffer", "#freeshipping",
    "#deals", "#sale", "#discount", "#bestprice",
    "#onlineshopping", "#buynow", "#musthave",
    "#techdeals", "#techsale", "#gadgetdeals"
]

# Product Image Styles for Instagram
IMAGE_STYLES = {
    "studio_shot": "Clean, minimalist studio setup, white background, soft studio lighting",
    "lifestyle": "Modern lifestyle setting, natural lighting, product in use",
    "unboxing": "Clean table setup, product being unboxed, close-up angle",
    "comparison": "Side-by-side product comparison, clean background, even lighting"
}

# Cost Tracking Constants
OPENAI_COSTS = {
    "gpt_4o_mini": 0.00015,  # $0.15 per 1K tokens
    "gpt_4o": 0.0025           # $2.50 per 1K tokens
}

MIDJOURNEY_COSTS = {
    "standard": 0.05,            # $0.05 per image
    "hd": 0.10,                # $0.10 per image
    "4k": 0.15                 # $0.15 per image
}

# Account Configuration Constants
ACCOUNT_CONFIGS = {
    "account_1": {
        "id": 1,
        "username": "",  # Will be loaded from env
        "primary_category": "phone_case",
        "secondary_categories": ["phone_film"]
    },
    "account_2": {
        "id": 2,
        "username": "",  # Will be loaded from env
        "primary_category": "earbuds",
        "secondary_categories": ["noise_cancelling_headphone"]
    }
}

SHARED_CATEGORIES = ["charging_cable", "charger"]

# Interaction Schedule Patterns
INTERACTION_SCHEDULE = {
    "morning": {"time_range": (9, 11), "percentage": 0.30},  # 30% of interactions
    "afternoon": {"time_range": (14, 16), "percentage": 0.30},  # 30% of interactions
    "evening": {"time_range": (20, 22), "percentage": 0.40}   # 40% of interactions
}

# Interaction Type Distribution
INTERACTION_TYPES = {
    "like": 0.60,      # 60% likes
    "comment": 0.30,    # 30% comments
    "follow": 0.10      # 10% follows
}

# E-commerce Caption Templates
CAPTION_TEMPLATES = {
    "hook_templates": [
        "🔥 Finally found the perfect {category}!",
        "⚡ This {category} is a game-changer",
        "✨ Meet your new favorite {category}",
        "💎 Premium quality at an unbeatable price",
        "🎯 Best {category} for your needs",
        "🚀 Level up with this {category}",
        "⭐ 5-star quality {category}"
    ],
    "cta_templates": [
        "👇 Shop link in bio!",
        "🔗 Link in bio - Limited stock!",
        "💰 Use code SAVE10 for 10% OFF",
        "📦 Free shipping worldwide!",
        "⚡ Order now - Ships in 24h!",
        "💡 Don't miss this deal - Link in bio",
        "🎁 Perfect gift idea! Link in bio"
    ]
}

"""
store/management/commands/seed_clean_store.py

Seeds the store database with:
- Exactly 10 clean, realistic e-commerce categories
- Exactly 30 high-end products (3 per category) with real titles and verified Unsplash photography
- Exactly 5 customer personas
- Clean zero reviews and zero orders
- Precomputed recommendations tailored to each persona's interests
"""

from decimal import Decimal
from django.core.management.base import BaseCommand
from django.utils import timezone
from store.models import Category, Customer, Product, Recommendation, Cart, CartItem, Order, OrderItem, Review


CATEGORIES_DATA = [
    {
        "name": "informatica_acessorios",
        "name_translated": "Computers & Accessories",
    },
    {
        "name": "telefonia",
        "name_translated": "Smartphones & Tablets",
    },
    {
        "name": "audio",
        "name_translated": "Audio & Headphones",
    },
    {
        "name": "relogios_presentes",
        "name_translated": "Watches & Wearables",
    },
    {
        "name": "cameras_foto",
        "name_translated": "Cameras & Photography",
    },
    {
        "name": "games",
        "name_translated": "Gaming & Consoles",
    },
    {
        "name": "esporte_lazer",
        "name_translated": "Sports & Outdoors",
    },
    {
        "name": "moveis_decoracao",
        "name_translated": "Furniture & Home Decor",
    },
    {
        "name": "eletrodomesticos",
        "name_translated": "Home Appliances",
    },
    {
        "name": "beleza_saude",
        "name_translated": "Beauty & Personal Care",
    },
]

PRODUCTS_DATA = [
    # 1. Computers & Accessories
    {
        "external_id": "prod_macbook_air_m2_001",
        "category_slug": "informatica_acessorios",
        "name": "Apple MacBook Air 13.6\" Chip M2 256GB Cinza-Espacial",
        "name_translated": "Apple MacBook Air 13.6\" M2 Chip 256GB SSD Space Gray",
        "price": Decimal("7499.00"),
        "avg_sentiment_score": 0.96,
        "total_purchases": 85,
        "image_url": "https://images.unsplash.com/photo-1517336714731-489689fd1ca8?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_dell_xps_15_002",
        "category_slug": "informatica_acessorios",
        "name": "Notebook Dell XPS 15 InfinityEdge Intel i7 16GB SSD 512GB",
        "name_translated": "Dell XPS 15 Laptop Core i7 16GB RAM 512GB SSD",
        "price": Decimal("8999.00"),
        "avg_sentiment_score": 0.91,
        "total_purchases": 42,
        "image_url": "https://images.unsplash.com/photo-1593642632823-8f785ba67e45?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_logitech_mx_mech_003",
        "category_slug": "informatica_acessorios",
        "name": "Teclado Mecânico Sem Fio Logitech MX Mechanical Mini",
        "name_translated": "Logitech MX Mechanical Mini Wireless Illuminated Keyboard",
        "price": Decimal("799.90"),
        "avg_sentiment_score": 0.94,
        "total_purchases": 134,
        "image_url": "https://images.unsplash.com/photo-1587829741301-dc798b83add3?auto=format&fit=crop&w=800&q=80",
    },

    # 2. Smartphones & Tablets
    {
        "external_id": "prod_iphone_15_pro_004",
        "category_slug": "telefonia",
        "name": "Apple iPhone 15 Pro Max 256GB Titânio Natural",
        "name_translated": "Apple iPhone 15 Pro Max 256GB Natural Titanium",
        "price": Decimal("8299.00"),
        "avg_sentiment_score": 0.97,
        "total_purchases": 210,
        "image_url": "https://images.unsplash.com/photo-1695048133142-1a20484d2569?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_galaxy_s24_ultra_005",
        "category_slug": "telefonia",
        "name": "Samsung Galaxy S24 Ultra 5G 512GB Titânio Preto",
        "name_translated": "Samsung Galaxy S24 Ultra 5G 512GB Titanium Black",
        "price": Decimal("7499.00"),
        "avg_sentiment_score": 0.93,
        "total_purchases": 168,
        "image_url": "https://images.unsplash.com/photo-1610945265064-0e34e5519bbf?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_ipad_air_m2_006",
        "category_slug": "telefonia",
        "name": "Apple iPad Air 11\" Chip M2 Wi-Fi 128GB Estelar",
        "name_translated": "Apple iPad Air 11-inch M2 Chip Wi-Fi 128GB Starlight",
        "price": Decimal("5499.00"),
        "avg_sentiment_score": 0.94,
        "total_purchases": 95,
        "image_url": "https://images.unsplash.com/photo-1544244015-0df4b3ffc6b0?auto=format&fit=crop&w=800&q=80",
    },

    # 3. Audio & Headphones
    {
        "external_id": "prod_sony_wh1000xm5_007",
        "category_slug": "audio",
        "name": "Headphone Sem Fio Sony WH-1000XM5 Cancelamento de Ruído",
        "name_translated": "Sony WH-1000XM5 Wireless Noise-Canceling Headphones",
        "price": Decimal("2199.00"),
        "avg_sentiment_score": 0.96,
        "total_purchases": 312,
        "image_url": "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_airpods_pro_2_008",
        "category_slug": "audio",
        "name": "Apple AirPods Pro (2ª geração) Estojo MagSafe USB-C",
        "name_translated": "Apple AirPods Pro 2nd Gen with MagSafe USB-C Case",
        "price": Decimal("1899.00"),
        "avg_sentiment_score": 0.95,
        "total_purchases": 420,
        "image_url": "https://images.unsplash.com/photo-1600294037681-c80b4cb5b434?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_marshall_emberton_009",
        "category_slug": "audio",
        "name": "Caixa de Som Bluetooth Marshall Emberton II Portátil",
        "name_translated": "Marshall Emberton II Portable Bluetooth Speaker",
        "price": Decimal("1199.00"),
        "avg_sentiment_score": 0.91,
        "total_purchases": 110,
        "image_url": "https://images.unsplash.com/photo-1545454675-3531b543be5d?auto=format&fit=crop&w=800&q=80",
    },

    # 4. Watches & Wearables
    {
        "external_id": "prod_apple_watch_ultra_010",
        "category_slug": "relogios_presentes",
        "name": "Apple Watch Ultra 2 GPS + Cellular Titânio 49mm",
        "name_translated": "Apple Watch Ultra 2 GPS + Cellular 49mm Titanium Case",
        "price": Decimal("6999.00"),
        "avg_sentiment_score": 0.97,
        "total_purchases": 78,
        "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_garmin_fenix_7_011",
        "category_slug": "relogios_presentes",
        "name": "Smartwatch Garmin Fenix 7 Pro Solar Sapphire Titânio",
        "name_translated": "Garmin Fenix 7 Pro Solar Sapphire Multisport GPS Watch",
        "price": Decimal("5499.00"),
        "avg_sentiment_score": 0.93,
        "total_purchases": 56,
        "image_url": "https://images.unsplash.com/photo-1524805444758-089113d48a6d?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_seiko_5_sports_012",
        "category_slug": "relogios_presentes",
        "name": "Relógio Automático Seiko 5 Sports Masculino Aço Inox",
        "name_translated": "Seiko 5 Sports Automatic Stainless Steel Men's Watch",
        "price": Decimal("1890.00"),
        "avg_sentiment_score": 0.89,
        "total_purchases": 64,
        "image_url": "https://images.unsplash.com/photo-1522335789203-aabd1fc54bc9?auto=format&fit=crop&w=800&q=80",
    },

    # 5. Cameras & Photography
    {
        "external_id": "prod_sony_a7_iv_013",
        "category_slug": "cameras_foto",
        "name": "Câmera Mirrorless Sony Alpha a7 IV Full-Frame (Corpo)",
        "name_translated": "Sony Alpha a7 IV Full-Frame Mirrorless Camera Body",
        "price": Decimal("14999.00"),
        "avg_sentiment_score": 0.98,
        "total_purchases": 38,
        "image_url": "https://images.unsplash.com/photo-1516035069371-29a1b244cc32?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_fujifilm_xt5_014",
        "category_slug": "cameras_foto",
        "name": "Câmera Mirrorless Fujifilm X-T5 Lente 18-55mm f/2.8-4",
        "name_translated": "Fujifilm X-T5 Mirrorless Camera with 18-55mm Lens Kit",
        "price": Decimal("11499.00"),
        "avg_sentiment_score": 0.94,
        "total_purchases": 49,
        "image_url": "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_dji_mini_4_015",
        "category_slug": "cameras_foto",
        "name": "Drone DJI Mini 4 Pro Fly More Combo Controle RC 2",
        "name_translated": "DJI Mini 4 Pro Fly More Combo Drone with RC 2 Controller",
        "price": Decimal("7899.00"),
        "avg_sentiment_score": 0.95,
        "total_purchases": 82,
        "image_url": "https://images.unsplash.com/photo-1527977966376-1c8408f9f108?auto=format&fit=crop&w=800&q=80",
    },

    # 6. Gaming & Consoles
    {
        "external_id": "prod_ps5_slim_016",
        "category_slug": "games",
        "name": "Console Sony PlayStation 5 Slim Edição Digital 1TB",
        "name_translated": "Sony PlayStation 5 Slim Digital Edition 1TB Console",
        "price": Decimal("3499.00"),
        "avg_sentiment_score": 0.96,
        "total_purchases": 380,
        "image_url": "https://images.unsplash.com/photo-1606813907291-d86efa9b94db?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_switch_oled_017",
        "category_slug": "games",
        "name": "Console Nintendo Switch OLED 64GB com Joy-Con Branco",
        "name_translated": "Nintendo Switch OLED Model 64GB with White Joy-Con",
        "price": Decimal("2199.00"),
        "avg_sentiment_score": 0.93,
        "total_purchases": 290,
        "image_url": "https://images.unsplash.com/photo-1578301978693-85fa9c0320b9?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_xbox_controller_018",
        "category_slug": "games",
        "name": "Controle Sem Fio Xbox Series X Carbon Black",
        "name_translated": "Xbox Wireless Controller Carbon Black for Series X/S & PC",
        "price": Decimal("429.00"),
        "avg_sentiment_score": 0.90,
        "total_purchases": 410,
        "image_url": "https://images.unsplash.com/photo-1600080972464-8e5f35f63d08?auto=format&fit=crop&w=800&q=80",
    },

    # 7. Sports & Outdoors
    {
        "external_id": "prod_nike_pegasus_019",
        "category_slug": "esporte_lazer",
        "name": "Tênis Nike Air Zoom Pegasus 40 Corrida Masculino",
        "name_translated": "Nike Air Zoom Pegasus 40 Men's Road Running Shoes",
        "price": Decimal("699.90"),
        "avg_sentiment_score": 0.93,
        "total_purchases": 340,
        "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_north_face_borealis_020",
        "category_slug": "esporte_lazer",
        "name": "Mochila The North Face Borealis 28L Resistente à Água",
        "name_translated": "The North Face Borealis 28L Water-Resistant Backpack",
        "price": Decimal("749.00"),
        "avg_sentiment_score": 0.92,
        "total_purchases": 195,
        "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_stanley_bottle_021",
        "category_slug": "esporte_lazer",
        "name": "Garrafa Térmica Stanley Quick Flip Go 700ml Inox",
        "name_translated": "Stanley Quick Flip GO Stainless Steel Water Bottle 700ml",
        "price": Decimal("249.00"),
        "avg_sentiment_score": 0.94,
        "total_purchases": 520,
        "image_url": "https://images.unsplash.com/photo-1602143407151-7111542de6e8?auto=format&fit=crop&w=800&q=80",
    },

    # 8. Furniture & Home Decor
    {
        "external_id": "prod_herman_miller_aeron_022",
        "category_slug": "moveis_decoracao",
        "name": "Cadeira Ergonômica de Escritório Herman Miller Aeron",
        "name_translated": "Herman Miller Aeron Ergonomic Office Desk Chair",
        "price": Decimal("8499.00"),
        "avg_sentiment_score": 0.98,
        "total_purchases": 65,
        "image_url": "https://images.unsplash.com/photo-1589384267710-7a170981ca78?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_oak_coffee_table_023",
        "category_slug": "moveis_decoracao",
        "name": "Mesa de Centro Minimalista em Madeira Maciça de Carvalho",
        "name_translated": "Minimalist Solid Oak Wood Coffee Table for Living Room",
        "price": Decimal("1290.00"),
        "avg_sentiment_score": 0.89,
        "total_purchases": 72,
        "image_url": "https://images.unsplash.com/photo-1533090481720-856c6e3c1fdc?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_arc_floor_lamp_024",
        "category_slug": "moveis_decoracao",
        "name": "Luminária de Chão Nórdica Moderna em Arco Dourado",
        "name_translated": "Modern Nordic Arc Floor Lamp Brass Finish with Marble Base",
        "price": Decimal("699.00"),
        "avg_sentiment_score": 0.90,
        "total_purchases": 88,
        "image_url": "https://images.unsplash.com/photo-1507473885765-e6ed057f782c?auto=format&fit=crop&w=800&q=80",
    },

    # 9. Home Appliances
    {
        "external_id": "prod_dyson_v15_detect_025",
        "category_slug": "eletrodomesticos",
        "name": "Aspirador de Pó Sem Fio Dyson V15 Detect Absolute",
        "name_translated": "Dyson V15 Detect Absolute Cordless Stick Vacuum Cleaner",
        "price": Decimal("5499.00"),
        "avg_sentiment_score": 0.96,
        "total_purchases": 115,
        "image_url": "https://images.unsplash.com/photo-1558317374-067fb5f30001?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_delonghi_dedica_026",
        "category_slug": "eletrodomesticos",
        "name": "Cafeteira Expresso Manual De'Longhi Dedica Deluxe Inox",
        "name_translated": "De'Longhi Dedica Deluxe 15-Bar Pump Espresso Machine",
        "price": Decimal("1499.00"),
        "avg_sentiment_score": 0.93,
        "total_purchases": 180,
        "image_url": "https://images.unsplash.com/photo-1517668808822-9ebb02f2a0e6?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_philips_airfryer_027",
        "category_slug": "eletrodomesticos",
        "name": "Fritadeira Elétrica Sem Óleo Philips Walita Airfryer 4.1L",
        "name_translated": "Philips Premium Digital Airfryer 4.1L Capacity",
        "price": Decimal("599.00"),
        "avg_sentiment_score": 0.95,
        "total_purchases": 390,
        "image_url": "https://images.unsplash.com/photo-1585515320310-259814833e62?auto=format&fit=crop&w=800&q=80",
    },

    # 10. Beauty & Personal Care
    {
        "external_id": "prod_dyson_supersonic_028",
        "category_slug": "beleza_saude",
        "name": "Secador de Cabelo Dyson Supersonic com Controle Térmico",
        "name_translated": "Dyson Supersonic Hair Dryer with Intelligent Heat Control",
        "price": Decimal("3299.00"),
        "avg_sentiment_score": 0.97,
        "total_purchases": 145,
        "image_url": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_braun_series_9_029",
        "category_slug": "beleza_saude",
        "name": "Barbeador Elétrico Braun Series 9 Pro Wet & Dry",
        "name_translated": "Braun Series 9 Pro Electric Shaver with SmartCare Center",
        "price": Decimal("1899.00"),
        "avg_sentiment_score": 0.93,
        "total_purchases": 98,
        "image_url": "https://images.unsplash.com/photo-1621607512214-68297480165e?auto=format&fit=crop&w=800&q=80",
    },
    {
        "external_id": "prod_philips_sonicare_030",
        "category_slug": "beleza_saude",
        "name": "Escova de Dentes Elétrica Philips Sonicare DiamondClean 9000",
        "name_translated": "Philips Sonicare DiamondClean 9000 Smart Electric Toothbrush",
        "price": Decimal("999.00"),
        "avg_sentiment_score": 0.94,
        "total_purchases": 160,
        "image_url": "https://images.unsplash.com/photo-1559656914-a30970c1affd?auto=format&fit=crop&w=800&q=80",
    },
]

CUSTOMERS_DATA = [
    {
        "external_id": "usr_alice_tech_01",
        "city": "São Paulo",
        "state": "SP",
        "fav_categories": ["informatica_acessorios", "telefonia", "audio"],
    },
    {
        "external_id": "usr_bruno_sports_02",
        "city": "Rio de Janeiro",
        "state": "RJ",
        "fav_categories": ["esporte_lazer", "relogios_presentes", "audio"],
    },
    {
        "external_id": "usr_camila_home_03",
        "city": "Curitiba",
        "state": "PR",
        "fav_categories": ["moveis_decoracao", "eletrodomesticos", "beleza_saude"],
    },
    {
        "external_id": "usr_diego_gaming_04",
        "city": "Belo Horizonte",
        "state": "MG",
        "fav_categories": ["games", "informatica_acessorios", "audio"],
    },
    {
        "external_id": "usr_elena_beauty_05",
        "city": "Salvador",
        "state": "BA",
        "fav_categories": ["beleza_saude", "cameras_foto", "relogios_presentes"],
    },
]


class Command(BaseCommand):
    help = "Destroys old database content and rebuilds with 10 clean categories, 30 real products with images, and 5 users."

    def handle(self, *args, **options):
        self.stdout.write("Wiping old data from all store tables...")
        Review.objects.all().delete()
        OrderItem.objects.all().delete()
        Order.objects.all().delete()
        CartItem.objects.all().delete()
        Cart.objects.all().delete()
        Recommendation.objects.all().delete()
        Product.objects.all().delete()
        Category.objects.all().delete()
        Customer.objects.all().delete()

        # 1. Insert 10 Categories
        self.stdout.write("Creating 10 clean categories...")
        cat_map = {}
        for cat_info in CATEGORIES_DATA:
            cat = Category.objects.create(
                name=cat_info["name"],
                name_translated=cat_info["name_translated"],
            )
            cat_map[cat.name] = cat
        self.stdout.write(f"  Created {len(cat_map)} categories.")

        # 2. Insert 30 Products
        self.stdout.write("Creating 30 real products with photography...")
        prod_map = {}
        for p_info in PRODUCTS_DATA:
            cat = cat_map[p_info["category_slug"]]
            prod = Product.objects.create(
                external_id=p_info["external_id"],
                category=cat,
                name=p_info["name"],
                name_translated=p_info["name_translated"],
                price=p_info["price"],
                avg_sentiment_score=p_info["avg_sentiment_score"],
                total_purchases=p_info["total_purchases"],
                image_url=p_info["image_url"],
            )
            prod_map[prod.external_id] = prod
        self.stdout.write(f"  Created {len(prod_map)} products.")

        # 3. Insert 5 Customers
        self.stdout.write("Creating 5 customer personas...")
        cust_map = {}
        for c_info in CUSTOMERS_DATA:
            cust = Customer.objects.create(
                external_id=c_info["external_id"],
                city=c_info["city"],
                state=c_info["state"],
            )
            cust_map[cust.external_id] = cust
        self.stdout.write(f"  Created {len(cust_map)} customers.")

        # 4. Generate high-quality personalized recommendations for each of the 5 users
        self.stdout.write("Precomputing personalized recommendations for each customer...")
        recs_count = 0
        now = timezone.now()
        for c_info in CUSTOMERS_DATA:
            cust = cust_map[c_info["external_id"]]
            fav_cats = c_info["fav_categories"]
            # Prioritize products in user's favorite categories
            matching_products = [
                p for p in prod_map.values() if p.category.name in fav_cats
            ]
            other_products = [
                p for p in prod_map.values() if p.category.name not in fav_cats
            ]
            ranked = matching_products[:6] + other_products[:2]

            score = 4.88
            for p in ranked:
                Recommendation.objects.create(
                    customer=cust,
                    product=p,
                    score=score,
                    generated_at=now,
                )
                score -= 0.09
                recs_count += 1

        self.stdout.write(f"  Created {recs_count} recommendations.")

        self.stdout.write(
            self.style.SUCCESS(
                f"\nSuccessfully seeded clean database:\n"
                f"  - Categories: {Category.objects.count()}\n"
                f"  - Products:   {Product.objects.count()}\n"
                f"  - Customers:  {Customer.objects.count()}\n"
                f"  - Reviews:    {Review.objects.count()} (0 reviews as requested)\n"
                f"  - Orders:     {Order.objects.count()} (0 orders as requested)\n"
                f"  - Recs:       {Recommendation.objects.count()}\n"
            )
        )

"""
MILGRAU Suite - Web Publisher (LIMP)
Collects Level 1 graphics (.webp), uploads them to Cloudflare R2,
generates static HTML Dashboards per year pointing to the cloud CDN,
and automatically updates the interactive measurement calendar.

@author: Fábio J. S. Lopes, Alexandre C. Yoshida, Alexandre Cacheffo, Luisa Mello
"""

import os
import sys
import glob
import re
import boto3
from datetime import datetime

# Import MILGRAU core functions
from functions.core_io import load_config, setup_logger, ensure_directories

# ==========================================
# CLOUDFLARE R2 SETUP & CREDENTIALS
# ==========================================
def get_cloud_credentials(logger):
    """Safely loads R2 credentials and initializes the boto3 client."""
    try:
        import credentials
        s3_client = boto3.client('s3',
            endpoint_url=credentials.R2_ENDPOINT,
            aws_access_key_id=credentials.R2_ACCESS_KEY,
            aws_secret_access_key=credentials.R2_SECRET_KEY,
            region_name='auto' 
        )
        return s3_client, credentials.R2_BUCKET_NAME, credentials.R2_PUBLIC_URL
    except ImportError:
        logger.critical("'credentials.py' not found! Please create it with your R2 keys. Exiting.")
        sys.exit(1)
    except AttributeError as e:
        logger.critical(f"Missing required variable in credentials.py: {e}. Exiting.")
        sys.exit(1)

def upload_to_r2(s3_client, bucket_name, local_file_path, cloud_file_key, logger):
    """Uploads a single file to the Cloudflare R2 Bucket."""
    try:
        s3_client.upload_file(local_file_path, bucket_name, cloud_file_key)
        return True
    except Exception as e:
        logger.error(f"  -> [R2 UPLOAD ERROR] Failed to upload {local_file_path}: {e}")
        return False

# ==========================================
# HTML DASHBOARD GENERATOR
# ==========================================
def generate_html_dashboard(html_path, prefix, date_title, valid_channels, valid_alts, has_global_mean, mean_rcs_file, year, cloud_public_url):
    """Generates the static HTML dashboard embedding cloud images."""
    default_ch = valid_channels[0] if valid_channels else ""
    default_alt = valid_alts[0] if valid_alts else ""
    
    channel_buttons = "".join([f'<button class="tab-btn ch-btn" onclick="setChannel(\'{ch}\', this)">{ch.replace("_", " ")}</button>\n' for ch in valid_channels])
    altitude_buttons = "".join([f'<button class="tab-btn alt-btn" onclick="setAltitude(\'{alt}\', this)">{alt} km</button>\n' for alt in valid_alts])

    global_tab_style = "display: inline-block;" if has_global_mean else "display: none;"
    cloud_base_url = f"{cloud_public_url}/{year}"

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Lidar Measurements - {date_title}</title>
  <style type="text/css">
    html, body {{ background: #f4f4f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; text-align: center; }}
    h2 {{ color: #333; margin-top: 20px; }}
    .subtitle {{ font-size: 18px; font-weight: normal; color: #666; display: block; margin-top: 5px; }}
    .dashboard {{ max-width: 100%; margin: 0 auto; padding: 20px; display: flex; flex-direction: column; align-items: center; }}
    .main-tabs {{ margin-bottom: 25px; }}
    .main-tab-btn {{ background: transparent; border: none; font-size: 18px; font-weight: bold; color: #777; cursor: pointer; padding: 10px 20px; margin: 0 10px; border-bottom: 3px solid transparent; transition: color 0.3s; }}
    .main-tab-btn:hover {{ color: #0056b3; }}
    .main-tab-btn.active {{ color: #0056b3; border-bottom: 3px solid #0056b3; }}
    .controls {{ display: flex; justify-content: center; gap: 40px; margin-bottom: 25px; flex-wrap: wrap; }}
    .tab-group {{ background: white; padding: 15px 25px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    .tab-group h3 {{ margin: 0 0 15px 0; font-size: 16px; color: #555; text-transform: uppercase; letter-spacing: 1px; }}
    .tab-btn {{ background: #e0e0e0; border: none; padding: 10px 20px; margin: 0 5px; border-radius: 5px; cursor: pointer; font-size: 15px; font-weight: bold; color: #333; transition: background 0.2s, transform 0.1s; }}
    .tab-btn:hover {{ background: #d0d0d0; transform: translateY(-2px); }}
    .tab-btn.active {{ background: #0056b3; color: white; box-shadow: 0 4px 8px rgba(0,86,179,0.3); }}
    .image-display {{ display: flex; flex-direction: column; align-items: center; gap: 20px; width: 100%; }}
    .image-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); width: 100%; max-width: 80%; transition: max-width 0.4s ease-in-out; }}
    .image-card img {{ width: 100%; height: auto; cursor: zoom-in; border-radius: 4px; display: block; }}
    #myModal {{ display: none; position: fixed; z-index: 100; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.85); }}
    .modal-content {{ margin: auto; display: block; width: 95%; max-width: 1600px; margin-top: 2%; animation: zoom 0.3s ease-in-out; }}
    @keyframes zoom {{ from {{transform:scale(0)}} to {{transform:scale(1)}} }}
  </style>
</head>
<body>
  <h2>Lidar Measurements<br><span class="subtitle">{date_title}</span></h2>
  
  <div class="dashboard">
      <div class="main-tabs">
          <button class="main-tab-btn active" id="tab-quicklooks" onclick="setMode('quicklooks')">Quicklooks</button>
          <button class="main-tab-btn" id="tab-resumo" style="{global_tab_style}" onclick="setMode('resumo')">Global Summary</button>
      </div>
      <div id="controls-panel" class="controls">
          <div class="tab-group"><h3>CHANNEL</h3>{channel_buttons}</div>
          <div class="tab-group"><h3>ALTITUDE</h3>{altitude_buttons}</div>
      </div>
      <div class="image-display">
          <div class="image-card" id="img-card">
              <img id="main-display" src="{cloud_base_url}/Quicklook_{prefix}_{default_ch}_{default_alt}km.webp" onclick="openModal(this.src)" alt="Lidar Image">
          </div>
      </div>
  </div>
  <div id="myModal" onclick="closeModal()"><img class="modal-content" id="img01"></div>

  <script>
    var currentChannel = "{default_ch}";
    var currentAltitude = "{default_alt}";
    var prefix = "{prefix}";
    var currentMode = "quicklooks";
    var cloudBaseUrl = "{cloud_base_url}";

    document.addEventListener("DOMContentLoaded", function() {{
        var firstCh = document.querySelector(".ch-btn");
        var firstAlt = document.querySelector(".alt-btn");
        if(firstCh) firstCh.classList.add("active");
        if(firstAlt) firstAlt.classList.add("active");
    }});

    function updateImage() {{
        var imgElement = document.getElementById("main-display");
        if (currentMode === "quicklooks") {{
            imgElement.src = cloudBaseUrl + "/Quicklook_" + prefix + "_" + currentChannel + "_" + currentAltitude + "km.webp";
        }} else {{
            imgElement.src = cloudBaseUrl + "/{mean_rcs_file}";
        }}
    }}

    function setMode(mode) {{
        currentMode = mode;
        document.getElementById("tab-quicklooks").classList.remove("active");
        document.getElementById("tab-resumo").classList.remove("active");
        var imgCard = document.getElementById("img-card");
        if (mode === "quicklooks") {{
            document.getElementById("tab-quicklooks").classList.add("active");
            document.getElementById("controls-panel").style.display = "flex"; 
            imgCard.style.maxWidth = "80%"; 
        }} else {{
            document.getElementById("tab-resumo").classList.add("active");
            document.getElementById("controls-panel").style.display = "none"; 
            imgCard.style.maxWidth = "50%"; 
        }}
        updateImage();
    }}

    function setChannel(ch, btnElement) {{
        currentChannel = ch;
        document.querySelectorAll(".ch-btn").forEach(btn => btn.classList.remove("active"));
        btnElement.classList.add("active");
        updateImage();
    }}

    function setAltitude(alt, btnElement) {{
        currentAltitude = alt;
        document.querySelectorAll(".alt-btn").forEach(btn => btn.classList.remove("active"));
        btnElement.classList.add("active");
        updateImage();
    }}

    var modal = document.getElementById("myModal");
    var modalImg = document.getElementById("img01");
    function openModal(src) {{ modal.style.display = "block"; modalImg.src = src; }}
    function closeModal() {{ modal.style.display = "none"; }}
  </script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# ==========================================
# CALENDAR INTEGRATION
# ==========================================
def update_calendar(base_site_folder, logger):
    """Scans the site folder and injects new entries into the JS calendar."""
    calendar_file = "ql-measurement-calendar.html"
    logger.info(f"Syncing interactive calendar ({calendar_file})...")
    
    calendar_path = os.path.join(base_site_folder, calendar_file)
    default_color = '#A3E4D7'
    
    if not os.path.exists(calendar_path):
        logger.warning(f"  -> [WARNING] Calendar file {calendar_file} not found in {base_site_folder}!")
        return

    with open(calendar_path, 'r', encoding='utf-8') as f:
        content = f.read()

    existing_urls = set(re.findall(r"url:\s*'(.*?)'", content))
    new_entries = ""
    counter = 0

    for year_folder in os.listdir(base_site_folder):
        year_path = os.path.join(base_site_folder, year_folder)
        
        if os.path.isdir(year_path) and year_folder.isdigit() and len(year_folder) == 4:
            for file in os.listdir(year_path):
                if file.endswith('_Dashboard.html') or file.endswith('_Gallery.html'):
                    relative_url = f"{year_folder}/{file}"
                    
                    if relative_url not in existing_urls:
                        match = re.match(r'^(\d{4})(\d{2})(\d{2})', file)
                        if match:
                            year = int(match.group(1))
                            js_month = int(match.group(2)) - 1 
                            day = int(match.group(3))
                            
                            new_entries += f"  {{\n    startDate: new Date({year}, {js_month}, {day}), endDate: new Date({year}, {js_month}, {day}), color: '{default_color}', url: '{relative_url}'\n  }},\n"
                            counter += 1

    if counter == 0:
        logger.info("  -> Calendar is up to date. No new measurements found.")
    else:
        logger.info(f"  -> Inserting {counter} new measurements into the calendar...")
        if "// MARCADOR_AUTOMATICO" in content:
            new_content = content.replace("// MARCADOR_AUTOMATICO", new_entries + "  // MARCADOR_AUTOMATICO")
            with open(calendar_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info("  -> [OK] Calendar successfully synced!")
        else:
            logger.error("  -> [ERROR] Missing '// MARCADOR_AUTOMATICO' tag in your HTML.")

# ==========================================
# MAIN ROUTINE (THE "VACUUM")
# ==========================================
if __name__ == "__main__":
    config = load_config()
    logger = setup_logger("LIMP", config['directories']['log_dir'])
    logger.info("=== Starting LIMP (Cloudflare R2 Upload & HTML Generation) ===")

    # Setup directories
    root_dir = os.getcwd() 
    base_data_folder = os.path.join(root_dir, config['directories']['level1_data'])
    base_site_folder = os.path.join(root_dir, config['directories']['site_output'])
    ensure_directories(base_site_folder)
    
    incremental = config['processing']['incremental']
    s3_client, bucket_name, cloud_public_url = get_cloud_credentials(logger)

    # 1. Vacuum Graphic Files
    search_pattern = os.path.join(base_data_folder, '**', '*.webp')
    all_images = glob.glob(search_pattern, recursive=True)
    
    if not all_images:
        logger.warning(f"No '.webp' images found in {config['directories']['level1_data']}. Exiting.")
        sys.exit(0)

    # 2. Aggregate Data by Measurement Date/Period
    measurements = {}
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        prefix = None
        
        if img_name.startswith("Quicklook_"):
            parts = img_name.replace(".webp", "").split("_")
            if len(parts) >= 5:
                prefix = parts[1]
                ch = f"{parts[2]}_{parts[3]}" 
                alt = parts[4].replace("km", "")
        elif img_name.startswith("MeanRCS_"):
            parts = img_name.replace(".webp", "").split("_")
            if len(parts) >= 2:
                prefix = parts[1]
            
        if prefix:
            if prefix not in measurements:
                measurements[prefix] = {'files': [], 'channels': set(), 'alts': set(), 'has_global_mean': False, 'mean_rcs_filename': ""}
            
            measurements[prefix]['files'].append(img_path)
            if img_name.startswith("Quicklook_"):
                measurements[prefix]['channels'].add(ch)
                measurements[prefix]['alts'].add(alt)
            elif img_name.startswith("MeanRCS_"):
                measurements[prefix]['has_global_mean'] = True
                measurements[prefix]['mean_rcs_filename'] = img_name

    # 3. Process Uploads and HTML
    processed_days = 0
    for prefix, data in measurements.items():
        try:
            year = prefix[:4]
            dt = datetime.strptime(prefix[:8], "%Y%m%d")
            date_str = dt.strftime("%d %b %Y")
        except ValueError:
            year, date_str = "Unknown", prefix

        site_year_folder = os.path.join(base_site_folder, year)
        ensure_directories(site_year_folder)
        html_path = os.path.join(site_year_folder, f"{prefix}_Dashboard.html")

        if incremental and os.path.exists(html_path):
            logger.debug(f"  -> [SKIPPED] Dashboard already exists for: {prefix}")
            continue
            
        logger.info(f"  -> [UPLOADING] Sending {len(data['files'])} images to Cloudflare R2 for {prefix}...")
        for img_path in data['files']:
            filename = os.path.basename(img_path)
            cloud_path = f"{year}/{filename}"
            upload_to_r2(s3_client, bucket_name, img_path, cloud_path, logger)
            
        valid_channels = sorted(list(data['channels']))
        valid_alts = sorted(list(data['alts']), key=lambda x: float(x) if x.replace('.','',1).isdigit() else 0)
        
        if valid_channels:
            generate_html_dashboard(
                html_path, prefix, date_str, valid_channels, valid_alts, 
                data['has_global_mean'], data['mean_rcs_filename'], year, cloud_public_url
            )
            logger.info(f"  -> [OK] HTML Dashboard created locally: {prefix}")
            processed_days += 1

    logger.info(f"=== LIMP Finished! {processed_days} new dashboards generated. ===")
    
    # 4. Sync Calendar
    update_calendar(base_site_folder, logger)
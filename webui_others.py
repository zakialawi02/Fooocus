import os
import shutil
import requests
from urllib.parse import urlparse, unquote

import gradio as gr

import modules.config
from modules.model_loader import load_file_from_url


def build_others_tab(
    prompt,
    negative_prompt,
    aspect_ratio,
    performance,
    styles,
    base_model,
    refiner_model,
    refiner_switch,
    image_number,
    lora_ctrls
):
    with gr.Tab(label="Others") as others_tab:
        with gr.Tab(label="Download"):
            file_input_path = gr.Textbox(
                label="File Path or URL",
                placeholder="Enter full path to file or downloadable URL",
                lines=1,
            )

            destination_folder = gr.Dropdown(
                label="Target Folder",
                choices=[
                    modules.config.paths_checkpoints[0],
                    modules.config.paths_loras[0],
                    modules.config.path_embeddings,
                    modules.config.path_vae,
                ],
                value=modules.config.paths_checkpoints[0],
            )

            download_result_text = gr.Textbox(
                label="Download Status", interactive=False
            )
            download_file_button = gr.Button(
                value="\u2b07 Download",
                variant="secondary",
                elem_classes="refresh_button",
            )

            def perform_download(file_url_or_path, target_directory):
                try:
                    if isinstance(target_directory, tuple):
                        target_directory = target_directory[1]

                    if file_url_or_path.startswith(("http://", "https://")):
                        response = requests.get(file_url_or_path, stream=True)
                        response.raise_for_status()
                        content_disposition = response.headers.get(
                            "Content-Disposition", ""
                        )
                        if "filename=" in content_disposition:
                            filename = content_disposition.split("filename=")[
                                -1
                            ].strip('"')
                        else:
                            parsed_url = urlparse(file_url_or_path)
                            filename = unquote(os.path.basename(parsed_url.path))
                        downloaded_path = load_file_from_url(
                            file_url_or_path,
                            model_dir=target_directory,
                            progress=True,
                            file_name=filename,
                        )
                        return f"\u2705 Downloaded to: {downloaded_path}"

                    if os.path.isfile(file_url_or_path):
                        filename = os.path.basename(file_url_or_path)
                        destination_path = os.path.join(target_directory, filename)
                        shutil.copy(file_url_or_path, destination_path)
                        return f"\u2705 Copied to: {destination_path}"

                    return "\u274c Error: File not found or invalid input."

                except Exception as e:
                    return f"\u274c Failed: {str(e)}"

            download_file_button.click(
                fn=perform_download,
                inputs=[file_input_path, destination_folder],
                outputs=[download_result_text],
            )

        with gr.Tab(label="Delete"):
            delete_folder_dropdown = gr.Dropdown(
                label="Select Folder",
                choices=[
                    modules.config.paths_checkpoints[0],
                    modules.config.paths_loras[0],
                    modules.config.path_embeddings,
                    modules.config.path_vae,
                    modules.config.get_dir_or_set_default("path_presets", "../presets"),
                ],
                value=modules.config.paths_checkpoints[0],
            )

            file_list_dropdown = gr.Dropdown(
                label="Select File to Delete", choices=[], multiselect=True
            )
            delete_button = gr.Button(
                value="\U0001f5d1 Delete Selected File(s)", variant="stop"
            )
            delete_status = gr.Textbox(
                visible=True, interactive=False, label="Delete Status"
            )

            def update_file_list(folder):
                try:
                    files = [
                        f
                        for f in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, f))
                    ]
                    return gr.update(choices=files, value=[])
                except Exception:
                    return gr.update(choices=[], value=[])

            def delete_selected_files(folder, selected_files):
                deleted = []
                errors = []

                for fname in selected_files:
                    try:
                        file_path = os.path.join(folder, fname)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            deleted.append(fname)
                        else:
                            errors.append(fname)
                    except Exception as e:
                        errors.append(f"{fname} (error: {e})")

                status = ""
                if deleted:
                    status += f"\u2705 Deleted: {', '.join(deleted)}. "
                if errors:
                    status += f"\u274c Failed: {', '.join(errors)}"
                if not deleted and not errors:
                    status = "\u26a0\ufe0f No files selected."

                try:
                    files = [
                        f
                        for f in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, f))
                    ]
                except Exception:
                    files = []

                return status.strip(), gr.update(choices=files, value=[])

            delete_folder_dropdown.change(
                update_file_list,
                inputs=[delete_folder_dropdown],
                outputs=[file_list_dropdown],
            )
            delete_button.click(
                delete_selected_files,
                inputs=[delete_folder_dropdown, file_list_dropdown],
                outputs=[delete_status, file_list_dropdown],
            )

        with gr.Tab(label="Backup/Restore"):
            with gr.Tab(label="Preset"):
                with gr.Row():
                    preset_name_input = gr.Textbox(
                        label="New Preset File Name",
                        placeholder="Enter a preset name (e.g., my_preset.json)",
                    )

                    preset_target_folder = gr.Dropdown(
                        label="Preset Target Folder",
                        choices=[
                            modules.config.get_dir_or_set_default(
                                "path_presets", "../presets"
                            )
                        ],
                        value=modules.config.get_dir_or_set_default(
                            "path_presets", "../presets"
                        ),
                    )

                preset_export_button = gr.Button(
                    value="\U0001f4e4 Export Preset", variant="secondary"
                )
                preset_export_status = gr.Textbox(
                    label="Preset Export Status", interactive=False
                )

                def export_preset(
                    prompt,
                    negative_prompt,
                    aspect_ratio,
                    performance,
                    styles,
                    base_model,
                    refiner_model,
                    refiner_switch,
                    image_number,
                    lora_ctrls,
                    preset_name,
                    preset_folder,
                ):
                    if not preset_name:
                        return "\u274c Please enter a valid preset name."

                    preset_data = {
                        "default_prompt": prompt,
                        "default_prompt_negative": negative_prompt,
                        "default_aspect_ratio": aspect_ratio,
                        "default_performance": performance,
                        "default_styles": styles,
                        "default_model": base_model,
                        "default_refiner": refiner_model,
                        "default_refiner_switch": refiner_switch,
                        "default_image_number": image_number,
                        "default_loras": [],
                    }

                    for i in range(0, len(lora_ctrls), 3):
                        enabled = lora_ctrls[i].value
                        name = lora_ctrls[i + 1].value
                        weight = lora_ctrls[i + 2].value
                        preset_data["default_loras"].append([enabled, name, weight])

                    preset_path = os.path.join(preset_folder, preset_name)
                    try:
                        import json
                        with open(preset_path, "w") as f:
                            json.dump(preset_data, f, indent=4)
                        return f"\u2705 Preset saved to: {preset_path}"
                    except Exception as e:
                        return f"\u274c Failed: {str(e)}"

                preset_export_button.click(
                    export_preset,
                    inputs=[
                        prompt,
                        negative_prompt,
                        aspect_ratio,
                        performance,
                        styles,
                        base_model,
                        refiner_model,
                        refiner_switch,
                        image_number,
                        lora_ctrls,
                        preset_name_input,
                        preset_target_folder,
                    ],
                    outputs=preset_export_status,
                )

            with gr.Tab(label="Backup"):
                with gr.Row():
                    backup_name = gr.Textbox(label="Backup Filename (no extension)", placeholder="e.g. my_config_backup")
                    backup_button = gr.Button(value="\u2B06 Backup Settings")
                    backup_file = gr.File(label="Download .json", interactive=False)

                def clean_aspect_ratio(value):
                    if not value:
                        return None
                    raw = value.split(" ")[0]
                    return raw.replace("×", "*")

                def backup_selected_settings(
                    filename,
                    prompt,
                    negative_prompt,
                    aspect_ratio,
                    performance,
                    styles,
                    base_model,
                    refiner_model,
                    refiner_switch,
                    image_number,
                    *lora_values
                ):
                    import json
                    if not filename:
                        return None

                    config = {
                        "default_prompt": prompt,
                        "default_prompt_negative": negative_prompt,
                        "default_aspect_ratio": clean_aspect_ratio(aspect_ratio),
                        "default_performance": performance,
                        "default_styles": styles,
                        "default_model": base_model,
                        "default_refiner": refiner_model,
                        "default_refiner_switch": refiner_switch,
                        "default_image_number": image_number,
                    }

                    loras = []
                    for i in range(0, len(lora_values), 3):
                        enabled = lora_values[i]
                        model = lora_values[i + 1]
                        weight = lora_values[i + 2]
                        if model and model != "None":
                            loras.append([enabled, model, weight])

                    if loras:
                        config["default_loras"] = loras

                    config = {k: v for k, v in config.items() if v not in [None, "", [], "None"]}

                    out_path = os.path.join("outputs", f"{filename}.json")
                    with open(out_path, "w") as f:
                        json.dump(config, f, indent=2)

                    return out_path

                backup_button.click(
                    fn=backup_selected_settings,
                    inputs=[
                        backup_name,
                        prompt,
                        negative_prompt,
                        aspect_ratio,
                        performance,
                        styles,
                        base_model,
                        refiner_model,
                        refiner_switch,
                        image_number,
                    ] + lora_ctrls,
                    outputs=[backup_file],
                )

            with gr.Tab(label="Restore"):
                with gr.Row():
                    restore_file = gr.File(label="Upload Config File (.json)", file_types=[".json"])
                    restore_button = gr.Button(value="\u267B️ Restore to Presets Folder")
                    restore_status = gr.Textbox(label="Status", interactive=False)

                def restore_config_file(file_obj):
                    try:
                        if file_obj is None:
                            return "\u26A0 No file selected."
                        filename = os.path.basename(file_obj.name)
                        presets_dir = os.path.join("presets")
                        os.makedirs(presets_dir, exist_ok=True)
                        destination = os.path.join(presets_dir, filename)
                        shutil.copy(file_obj.name, destination)
                        return f"\u2705 Saved to presets/{filename}"
                    except Exception as e:
                        return f"\u274C Failed to restore: {e}"

                restore_button.click(
                    fn=restore_config_file,
                    inputs=[restore_file],
                    outputs=[restore_status],
                )

            gr.Markdown(
                "You can backup your current settings and restore them later. "
                "This is useful for saving configurations or sharing with others."
            )

    others_tab.select(
        fn=update_file_list,
        inputs=[delete_folder_dropdown],
        outputs=[file_list_dropdown],
        queue=False,
        show_progress=False,
    )

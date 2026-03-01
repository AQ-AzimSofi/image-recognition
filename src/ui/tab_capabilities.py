import gradio as gr

from ..rekognition import RekognitionClient, RekognitionError


def _explore_capabilities(image_path, client):
    if image_path is None:
        return [], ""

    try:
        result = client.detect_labels_from_path(image_path, min_confidence=1.0, max_labels=100)
    except RekognitionError as e:
        return [], f"Error: {e}"

    table_data = []
    for label in result.labels:
        has_instance = "Yes" if label.instances else "No"
        categories = ", ".join(label.categories) if label.categories else "-"
        parents = ", ".join(label.parents) if label.parents else "-"
        table_data.append([
            label.name,
            f"{label.confidence:.1f}%",
            categories,
            parents,
            has_instance,
            len(label.instances),
        ])

    with_bbox = sum(1 for l in result.labels if l.instances)
    without_bbox = sum(1 for l in result.labels if not l.instances)

    all_categories = set()
    for l in result.labels:
        all_categories.update(l.categories)

    summary = (
        f"**Total labels detected:** {len(result.labels)}  \n"
        f"**With bounding box (physical objects):** {with_bbox}  \n"
        f"**Without bounding box (abstract/scene):** {without_bbox}  \n"
        f"**Categories found:** {', '.join(sorted(all_categories)) if all_categories else 'None'}"
    )

    return table_data, summary


def create_tab(client: RekognitionClient) -> gr.Blocks:
    with gr.Blocks() as tab:
        gr.Markdown("## Object Capabilities Explorer")
        gr.Markdown(
            "Runs detection at minimum confidence (1%) to discover everything "
            "Rekognition can detect in an image. Useful for understanding the label taxonomy."
        )

        with gr.Row():
            image_input = gr.Image(type="filepath", label="Upload Image")
            explore_btn = gr.Button("Detect Everything", variant="primary")

        summary_md = gr.Markdown("")

        results_table = gr.Dataframe(
            headers=["Label", "Confidence", "Category", "Parents", "Has Instance", "Instance Count"],
            label="All Detected Labels",
        )

        explore_btn.click(
            fn=lambda img: _explore_capabilities(img, client),
            inputs=[image_input],
            outputs=[results_table, summary_md],
        )

    return tab

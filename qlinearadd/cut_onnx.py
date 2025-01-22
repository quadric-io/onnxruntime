import click
import onnx

from tvm.contrib.epu import onnx_util


@click.command()
@click.argument("onnx-file", type=click.Path(exists=True))
@click.argument("output-file", type=click.Path(exists=False))
@click.argument("output-file2", type=click.Path(exists=False), required=False)
@click.option(
    "-b",
    "--cut-before",
    multiple=True,
    help="Names of ValueInfoProtos before which to cut the graph. Option can be used multiple times per command; supply one per use of this option",
)
@click.option(
    "-a",
    "--cut-after",
    multiple=True,
    help="Names of ValueInfoProtos after which to cut the graph. Option can be used multiple times per comamnd; supply one per use of this option",
)
@click.option(
    "-s",
    "--split-on",
    multiple=True,
    help="Names of ValueInfoProtos on which to split the graph. If this option is used, two ONNX models will be returned: one from the inputs to these nodes, and one from these nodes the the outputs. Option can be used multiple times per comamnd; supply one per use of this option",
)
def cut(onnx_file, output_file, output_file2, cut_before, cut_after, split_on):
    if bool(cut_before or cut_after) == bool(split_on):
        raise click.BadOptionUsage(
            split_on, "Invalid usage: supply either split_on, or cut-before/cut-after, but not both"
        )

    onnx_model = onnx.load(onnx_file)

    if cut_before or cut_after:
        onnx_model = onnx_util.cut_onnx(onnx_model, cut_before, cut_after)
        onnx.save(onnx_model, output_file)

    if split_on:
        if not output_file2:
            raise click.BadArgumentUsage(
                "If using -s/--split-on, please supply two output file paths"
            )
        model1, model2 = onnx_util.split_onnx(onnx_model, split_on)
        onnx.save(model1, output_file)
        onnx.save(model2, output_file2)


if __name__ == "__main__":
    cut()

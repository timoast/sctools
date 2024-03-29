import sys
from argparse import ArgumentParser
import pkg_resources
from sctools import cli


version = pkg_resources.require("sctools")[0].version
parser = ArgumentParser(description='Collection of tools for 10x chromium single-cell RNA-seq data analysis')
parser.add_argument('--version', action='version', version='%(prog)s '+str(version))
subparsers = parser.add_subparsers(title='Subcommands')

# filterbarcodes
parser_filterbarcodes = subparsers.add_parser('filterbarcodes', description='Filter reads based on input list of cell barcodes')
parser_filterbarcodes.add_argument('-b', '--bam', help='Input bam file (must be indexed)', required=True, type=str)
parser_filterbarcodes.add_argument('-c', '--cells', help='File or comma-separated list of cell barcodes. Can be gzip compressed', required=True, type=str)
parser_filterbarcodes.add_argument('-o', '--output', help='Name for output text file', required=True, type=str)
parser_filterbarcodes.add_argument('-t', '--trim_suffix', help='Remove trail 2 characters from cell barcode in BAM file', action='store_true', default=False)
parser_filterbarcodes.add_argument('-s', '--sam', help='Output sam format (default bam output)', required=False, action='store_true', default=False)
parser_filterbarcodes.add_argument('-p', '--nproc', help='Number of processors (default = 1)', required=False, default=1, type=int)
parser_filterbarcodes.set_defaults(func=cli.run_filterbarcodes)

# countsnps
parser_countsnps = subparsers.add_parser('countsnps', description='Count reference and alternate SNPs per cell in single-cell RNA data')
parser_countsnps.add_argument('-b', '--bam', help='Input bam file (must be indexed)', required=True, type=str)
parser_countsnps.add_argument('-s', '--snp', help='File with SNPs. Needs chromosome, position, reference, alternate as first four columns',
                              required=True, type=str)
parser_countsnps.add_argument('-o', '--output', help='Name for output text file', required=True, type=str)
parser_countsnps.add_argument('-c', '--cells', help='File or comma-separated list of cell barcodes to count SNPs for. Can be gzip compressed',
                              required=False, type=str)
parser_countsnps.add_argument('-p', '--nproc', help='Number of processors (default = 1)', required=False, default=1, type=int)
parser_countsnps.set_defaults(func=cli.run_countsnps)

# countedited
parser_countedited = subparsers.add_parser('countedited', description='Count edited transcripts per gene per cell in single-cell RNA data. Output is a matrix of positions by cells.')
parser_countedited.add_argument('-b', '--bam', help='Input bam file (must be indexed)', required=True, type=str)
parser_countedited.add_argument('-e', '--edit', help='File with edited base coordinates. Needs chromosome, position, reference, alternate as first four columns',
                                required=True, type=str)
parser_countedited.add_argument('-o', '--output', help='Name for output text file', required=True, type=str)
parser_countedited.add_argument('-c', '--cells', help='File containing cell barcodes to count edited bases for. Can be gzip compressed',
                                required=False, type=str)
parser_countedited.add_argument('-p', '--nproc', help='Number of processors (default = 1)', required=False, default=1, type=int)
parser_countedited.set_defaults(func=cli.run_countedited)

# genotype
parser_genotype = subparsers.add_parser('genotype', description='Genotype cells based on SNP UMI counts.')
parser_genotype.add_argument('-s', '--infile', help='SNP UMI counts for each genotype', required=True, type=str)
parser_genotype.add_argument('--sample_name', help='Sample name', required=True, type=str)
parser_genotype.add_argument('-n', '--nproc', help='Number of processors (default = 1)', required=False, default=1, type=int)
parser_genotype.add_argument('-p', '--plot', help='Plot results', required=False, default=False, action='store_true')
parser_genotype.add_argument('--summarize', help='Summarize results and write to file', required=False, default=False, action='store_true')
parser_genotype.add_argument('-d', '--downsample', help='Do not downsample cells before detecting background cluster',
                             required=False, default=True, action='store_false')
parser_genotype.add_argument('--max_difference', help='Maximum UMI count difference between genotypes before applying downsampling to equalize UMI count distribution (default = 0.2)',
                             required=False, default=0.2, type=float)
parser_genotype.add_argument('--eps_background', help='DBSCAN epsilon value for background detection (default = 0.5)',
                             required=False, default=0.5, type=float)
parser_genotype.add_argument('--eps_background_core', help='DBSCAN epsilon value for core background detection (default = 0.2)',
                             required=False, default=0.2, type=float)
parser_genotype.add_argument('--eps_cells', help='DBSCAN epsilon value for cell clustering (default = 0.2)',
                             required=False, default=0.2, type=float)
parser_genotype.add_argument('--eps_margin', help='DBSCAN epsilon value for margin cell clustering (default = 0.1)',
                             required=False, default=0.1, type=float)
parser_genotype.add_argument('--min_samples_background', help='Minimum number of cells in each cluster for background cell clustering (default = 300)',
                             required=False, default=300, type=int)
parser_genotype.add_argument('--min_samples_cells', help='Minimum number of cells in each cluster for cell clustering (default = 100)',
                             required=False, default=100, type=int)
parser_genotype.add_argument('--min_umi_total', help='Minimum total number of UMIs for a cell barcode to be entered into clustering steps (default = 20)',
                             required=False, default=20, type=int)
parser_genotype.add_argument('--min_umi_each', help='Minimum number of UMIs for each genotype for a cell barcode to be entered into clustering steps (default = 10)',
                             required=False, default=10, type=int)
parser_genotype.add_argument('--margin', help='Detect droplets on the margin between background and cells',
                             required=False, default=False, action='store_true')
parser_genotype.set_defaults(func=cli.run_genotyping)

parser_replace = subparsers.add_parser('replace', description='substitute SNPs into reference genome')
parser_replace.add_argument('-g', '--genome', help='reference genome fasta file', required=True, type=str)
parser_replace.add_argument('-s', '--snp', help='snp file in tsv format', required=True, type=str)
parser_replace.add_argument('-o', '--output', help='output filename', required=True, type=str)
parser_replace.set_defaults(func=cli.run_replace)

def main():
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()
    else:
        options = parser.parse_args()
        options.func(options)

if __name__ == "__main__":
    main()
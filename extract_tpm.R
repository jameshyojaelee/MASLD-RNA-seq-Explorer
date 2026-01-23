
tryCatch({
    txi <- readRDS("/gpfs/commons/groups/sanjana_lab/Cas13/RNA-seq/patient_RNAseq/results/GSE130970/salmon_merged/tximport_salmon.rds")
    print(names(txi))
    if ("abundance" %in% names(txi)) {
        print("Abundance (TPM) matrix found.")
        print(dim(txi$abundance))
        print(head(rownames(txi$abundance)))
        
        # Calculate mean TPM across all samples
        mean_tpm <- rowMeans(txi$abundance)
        write.csv(data.frame(gene_id=names(mean_tpm), tpm_mean=mean_tpm), "GSE130970_mean_tpm.csv", row.names=FALSE)
        print("Saved GSE130970_mean_tpm.csv")
    }
}, error = function(e) { print(e) })

tryCatch({
    txi <- readRDS("/gpfs/commons/groups/sanjana_lab/Cas13/RNA-seq/patient_RNAseq/results/GSE135251/salmon_merged/tximport_salmon.rds")
    if ("abundance" %in% names(txi)) {
        mean_tpm <- rowMeans(txi$abundance)
        write.csv(data.frame(gene_id=names(mean_tpm), tpm_mean=mean_tpm), "GSE135251_mean_tpm.csv", row.names=FALSE)
        print("Saved GSE135251_mean_tpm.csv")
    }
}, error = function(e) { print(e) })

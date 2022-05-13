## Packages
##   Following packages are needed for data manipulation:

## [[file:pr_analysis.org::*Packages][Packages:1]]
# install.packages("tidyverse")
library("dplyr")
library("purrr")
## Packages:1 ends here



## This package is needed for PCA plotting:

## [[file:pr_analysis.org::*Packages][Packages:2]]
# install.packages("ggfortify")
library("ggfortify")
## Packages:2 ends here



## Following package is used for printing tables into the PDF file:

## [[file:pr_analysis.org::*Packages][Packages:3]]
# install.packages("gridExtra")
library("gridExtra")
## Packages:3 ends here



## This library is used to format percentages:

## [[file:pr_analysis.org::*Packages][Packages:4]]
# install.packages("scales")
library("scales")
## Packages:4 ends here



## This package can be used to export tables into the LaTeX:

## [[file:pr_analysis.org::*Packages][Packages:5]]
# install.packages("xtable")
library("xtable")
## Packages:5 ends here



## Following package is able to export plots into the LaTeX (TikZ):

## [[file:pr_analysis.org::*Packages][Packages:6]]
# install.packages("tikzDevice")
library("tikzDevice")
## Packages:6 ends here



## This library is used to calculate effect size (Cramer's V):

## [[file:pr_analysis.org::*Packages][Packages:7]]
# install.packages("lsr")
library("lsr")
## Packages:7 ends here



## This is needed to be able to use variables instead of column names when using =dplyr=:

## [[file:pr_analysis.org::*Packages][Packages:8]]
# install.packages("lazyeval")
library("lazyeval")
## Packages:8 ends here

## Dataset loading
##    Load dataset with pull requests from the CSV file:

## [[file:pr_analysis.org::*Dataset loading][Dataset loading:1]]
import_prs <- function(path) {
    data <- read.csv(path, header=TRUE, sep = ",", check.names=FALSE)

    # Convert strings to booleans
    data$accepted <- data$accepted == "True"
    data$submitter_is_project_member <- data$submitter_is_project_member == "True"

    data
}

prs <- import_prs("data.csv")

dim(prs)

str(prs)

summary(prs)
## Dataset loading:1 ends here

## Filter linter results

## [[file:pr_analysis.org::*Filter linter results][Filter linter results:1]]
prsQuality <- prs %>% select("accepted" | starts_with('results_'))

summary(prsQuality)
## Filter linter results:1 ends here

## Research Question 1
##   Print the table with a summary of all projects:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:1]]
projects_summary <- (prs %>%
                     mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x < 0, 0L, .x))),
                            fixed=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x > 0, 0L, -.x))),
                            rejected=!accepted) %>% group_by(project_name) %>% rename(Project=project_name) %>%
                     summarise(Stars=first(project_number_of_watchers), "Analyzed PRs"=n(),
                               Accepted=percent(sum(accepted)/n()), Rejected=percent(sum(rejected)/n()),
                               "Introduced issues"=mean(introduced, trim=0.05),
                               "Fixed issues"=mean(fixed, trim=0.05)) %>% arrange(desc(Stars))
                     )

print(xtable(projects_summary, type="latex", align=c("l", "|p{4cm}", "p{0.8cm}", "p{1cm}", "p{1cm}",
                                                     "p{1cm}", "p{1.2cm}", "p{0.8cm}|")),
      file="projects_summary.tex", include.rownames=FALSE,
      add.to.row=list(pos=list(0), command=c("\\hline")), floating=FALSE)
## Research Question 1:1 ends here



## Print the scatter plot between number of stars and percentage of accepted PRs:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:2]]
tikz(filename="stars_and_acceptance.tex", width=8, height=3)
ggplot(projects_summary %>% mutate_at("Accepted", function(x) readr::parse_number(x)), aes(x=Stars, y=Accepted)) + geom_point() +
    geom_smooth(method='lm') + labs(y="Accepted PRs (\\%)")
dev.off()
## Research Question 1:2 ends here



## Print the heat map of fixed and introduced issues (PR quality overview):

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:3]]
tikz(filename="pr_quality_heat_map.tex", width=8, height=4)
ggplot(prs %>% mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x < 0, 0L, .x))),
                      fixed=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x > 0, 0L, -.x)))),
       aes(x=introduced, y=fixed)) + xlim(-11,210) + ylim(-11,210) + geom_bin2d(binwidth=10) +
    scale_fill_gradient(trans="log10") + labs(x="Introduced issues", y="Fixed issues",
                                              fill="Number\nof PRs")
dev.off()
## Research Question 1:3 ends here



## Print the summary about all pull request of the given language:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:4]]
prs %>% mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x < 0, 0L, .x))),
               fixed=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x > 0, 0L, -.x))),
               rejected=!accepted) %>%
    summarise(Stars=mean(project_number_of_watchers), "Analyzed PRs"=n(),
              Accepted=percent(sum(accepted)/n()), Rejected=percent(sum(rejected)/n()),
              "Introduced issues"=mean(introduced, trim=0.05), "Fixed issues"=mean(fixed, trim=0.05))
## Research Question 1:4 ends here



## Do the same but group by acceptance:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:5]]
prs %>% mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x < 0, 0L, .x))),
               fixed=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x > 0, 0L, -.x))),
               rejected=!accepted) %>% group_by(accepted) %>%
    summarise(Stars=mean(project_number_of_watchers), "Analyzed PRs"=n(),
              Accepted=percent(sum(accepted)/n()), Rejected=percent(sum(rejected)/n()),
              "Introduced issues"=mean(introduced, trim=0.05), "Fixed issues"=mean(fixed, trim=0.05)) %>%
    print(width = Inf)
## Research Question 1:5 ends here



## Print the number of pull requests that did not changed the quality:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:6]]
prs %>% mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x < 0, 0L, .x))),
               fixed=rowSums(mutate_all(select(., starts_with("results_")), ~if_else(.x > 0, 0L, -.x)))) %>%
    filter(introduced == 0, fixed == 0) %>% nrow
## Research Question 1:6 ends here



## Summarize information about individual issues (compute maximum, minimum etc.):

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:7]]
(issues <- prsQuality %>% group_by(accepted) %>% summarise(across(everything(),
                                                                  tibble::lst(max, min, mean, introduced_by=~sum(. > 0),
                                                                              fixed_by=~sum(. < 0), appeared_in=~sum(. != 0)),
                                                                 .names="{.col}***{.fn}")) %>%
        tidyr::pivot_longer(cols=starts_with("results_"), names_to=c("issue", ".value"), names_sep="\\*\\*\\*") %>%
        group_by(accepted) %>% group_split() %>% bind_cols() %>% select(2:8, 11:16) %>%
        rename_with(.cols=2:7, .fn=function(x) sub("^", "rejected.", sub("\\..*", "", x))) %>%
        rename_with(.cols=8:13, .fn=function(x) sub("^", "accepted.", sub("\\..*", "", x))) %>%
        rename(issue = issue...2) %>% mutate_at("issue", function(x) sub("results_([^_]+)_", "", x)) %>%
        tidyr::extract(issue, into=c("type", "issue"), "^([^_]+)_(.*)"))
## Research Question 1:7 ends here



## Print the table with summary into the PDF file:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:8]]
pdf("issues.pdf", height=75, width=25)
grid.table(issues)
dev.off()
## Research Question 1:8 ends here



## Print the number of different issue that was detected in the PRs:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:9]]
nrow(issues)
## Research Question 1:9 ends here



## Print the projects that introduced the issue:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:10]]
for (issue in issues$issue) {
    print(issue)
    column_name <- names(prs)[grep(paste("_", issue, sep=""), names(prs))]
    prs %>% filter_(interp(~v > 0, v=as.name(column_name))) %>% distinct(project_name) %>% print
}
## Research Question 1:10 ends here



## Summarize the issue categories:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:11]]
issueTypesSummary <- tibble(
  type = character(),
  introduced_total = integer(),
  introduced_by = integer(),
  fixed_total = integer(),
  fixed_by = integer()
)
for (type in unique(issues$type)) {
    issueTypesSummary <- issueTypesSummary %>%
        bind_rows(prs %>% select(starts_with("results_") & contains(type)) %>%
                  mutate(introduced=rowSums(mutate_all(., ~if_else(.x < 0, 0L, .x))),
                         fixed=rowSums(mutate_all(., ~if_else(.x > 0, 0L, -.x)))) %>%
                  summarize(type=type, introduced_total=sum(introduced), introduced_by=sum(introduced > 0),
                            fixed_total=sum(fixed), fixed_by=sum(fixed > 0)))
}

print(xtable((issueTypesSummary %>% rename(Category=type, "Introduced in total"=introduced_total,
                                           "Introduced by PR"=introduced_by, "Fixed\\newline{}in total"=fixed_total,
                                           "Fixed by PR"=fixed_by)),
             type="latex", align=c("l", "|p{2.5cm}", "p{1.2cm}", "p{1.2cm}", "p{1.2cm}", "p{0.8cm}|"), digits=c(0,0,0,0,0,0)),
      sanitize.text.function=identity, file="issue_types_summary.tex", include.rownames=FALSE,
      add.to.row=list(pos=list(0), command=c("\\hline")), floating=FALSE)
## Research Question 1:11 ends here



## Create a barplot with issues and their average counts in accepted/rejected pull requests:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:12]]
barplot(t(as.matrix(issues %>% select(accepted.mean, rejected.mean))), beside=TRUE, legend.text=TRUE,
        xlab="issue", ylab="on average in one PR")
## Research Question 1:12 ends here



## List the issues sorted by the number of pull request which introduced them:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:13]]
issues %>% mutate(introduced_by=accepted.introduced_by + rejected.introduced_by) %>%
    arrange(desc(introduced_by)) %>% select(type, issue, introduced_by)
## Research Question 1:13 ends here



## List the issues sorted by the number of pull request which fixed them:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:14]]
issues %>% mutate(fixed_by=accepted.fixed_by + rejected.fixed_by) %>%
    arrange(desc(fixed_by)) %>% select(type, issue, fixed_by)
## Research Question 1:14 ends here



## List the issues sorted by the number of accepted pull request which introduced them:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:15]]
issues %>% arrange(desc(accepted.introduced_by)) %>% select(type, issue, accepted.introduced_by)
## Research Question 1:15 ends here



## List the issues sorted by the number of rejected pull request which introduced them:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:16]]
issues %>% arrange(desc(rejected.introduced_by)) %>% select(type, issue, rejected.introduced_by)
## Research Question 1:16 ends here



## List the issues sorted by the number of accepted pull request which fixed them:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:17]]
issues %>% arrange(desc(accepted.fixed_by)) %>% select(type, issue, accepted.fixed_by)
## Research Question 1:17 ends here



## List the issues sorted by the number of rejected pull request which fixed them:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:18]]
issues %>% arrange(desc(rejected.fixed_by)) %>% select(type, issue, rejected.fixed_by)
## Research Question 1:18 ends here



## List the issues and the percentage in how many pull requests they change the quality:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:19]]
issues %>% transmute(type, issue, appeared_in=(rejected.appeared_in + accepted.appeared_in)) %>%
    arrange(desc(appeared_in)) %>% mutate(percent_of_prs=percent(appeared_in/nrow(prs))) %>%
    print(n=Inf)
## Research Question 1:19 ends here



## Print the issues that were fixed in the larger number of PRs then introduced.

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:20]]
issues %>% transmute(type, issue, fixed_more_times=(accepted.fixed_by + rejected.fixed_by -
                                                    accepted.introduced_by - rejected.introduced_by)) %>%
    arrange(desc(fixed_more_times)) %>% print(n=Inf)
## Research Question 1:20 ends here



## Create a barplot with issues on the x-axis and number of pull request in which the issues were fixed/introduced on the y-axis:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:21]]
tikz(filename="issues_appeared_in.tex", width=8, height=4)
issues %>% transmute(type, appeared_in=100*(rejected.appeared_in + accepted.appeared_in)/nrow(prs)) %>%
    arrange(desc(appeared_in)) %>% mutate(pos=1:n()) %>%
    ggplot(aes(x=pos, y=appeared_in, fill=type)) + geom_col() + labs(x="Issues", y="Pull Requests (\\%)", fill="Types") +
    theme(axis.ticks.x=element_blank(), axis.text.x=element_blank())
dev.off()
## Research Question 1:21 ends here



## Create a scatter plot with issue types on the x-axis and number of pull request in which the issues were
## fixed/introduced on the y-axis:

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:22]]
tikz(filename="issues_types_and_prs.tex", width=8, height=4)
issues %>% transmute(type, appeared_in=100*(rejected.appeared_in + accepted.appeared_in)/nrow(prs)) %>%
    ggplot(aes(x=reorder(type, desc(appeared_in), mean), y=appeared_in)) + labs(x="Issue types (sorted by y-axis mean)",
                                                                                y="Pull Requests (\\%)") + geom_point()
dev.off()
## Research Question 1:22 ends here



## Print the issue types and percentage of PRs that contained the average issue from the given category.

## [[file:pr_analysis.org::*Research Question 1][Research Question 1:23]]
issues %>% transmute(type, appeared_in=100*(rejected.appeared_in + accepted.appeared_in)/nrow(prs)) %>%
    group_by(type) %>% summarize(mean(appeared_in))
## Research Question 1:23 ends here

## Research Question 2
##   Import the issue importance from CSV files:

## [[file:pr_analysis.org::*Research Question 2][Research Question 2:1]]
import_issue_importance <- function(path) {
    (lapply(list.files(path=path, pattern="*.csv"),
           (function (file) read.csv(paste(path, file, sep=""), header=TRUE, sep = ",",
                                     check.names=FALSE) %>% rename_with(~sub("_ruleid.csv", "", file),
                                                                        Importance)))
    ) %>% reduce(full_join, by="Variables") %>%
          mutate_at("Variables", function(x) sub("results_([^_]+)_", "", x)) %>%
          tidyr::extract(Variables, into=c("type", "issue"), "^([^_]+)_(.*)")
}

issueImportance <- import_issue_importance("classification/Importances_drop/values/")

introducedIssueImportance <- import_issue_importance("classification_introduced/Importances_drop/values/")

fixedIssueImportance <- import_issue_importance("classification_fixed/Importances_drop/values/")
## Research Question 2:1 ends here



## Sort issues by their average importance and print them:

## [[file:pr_analysis.org::*Research Question 2][Research Question 2:2]]
issueImportance %>% mutate(mean=rowMeans(.[,-1:-2])) %>% arrange(desc(mean)) %>% head(10) %>% print

introducedIssueImportance %>% mutate(mean=rowMeans(.[,-1:-2])) %>% arrange(desc(mean)) %>% head(10) %>% print

fixedIssueImportance %>% mutate(mean=rowMeans(.[,-1:-2])) %>% arrange(desc(mean)) %>% head(10) %>% print
## Research Question 2:2 ends here



## Sort issues by their average importance and plot them in the barplot:

## [[file:pr_analysis.org::*Research Question 2][Research Question 2:3]]
plot_issue_importance <- function(issue_importance) {
    issue_importance %>% mutate(mean=rowMeans(.[,-1:-2])) %>% arrange(desc(mean)) %>% head(10) %>%
        tidyr::gather(classifier, importance, -c(type, issue, mean)) %>% ggplot() +
            geom_bar(aes(x=reorder(issue, mean), y=(100 * importance), fill=classifier), stat='identity',
                     position = "dodge", width=.7) +
            coord_flip() + labs(x=NULL, y="Importance (\\%)", fill="Classifier") + geom_hline(yintercept=0)
}

tikz(filename="issue_importance.tex", width=8, height=6)
plot_issue_importance(issueImportance)
dev.off()

plot_issue_importance(issueImportance)

plot_issue_importance(introducedIssueImportance)

plot_issue_importance(fixedIssueImportance)
## Research Question 2:3 ends here

## PCA scatterplot

## [[file:pr_analysis.org::*PCA scatterplot][PCA scatterplot:1]]
set.seed(135089)
prsSample <- prsQuality %>% sample_n(2000, replace=FALSE)

acceptancePCA <- prcomp(prsSample %>% select(-accepted))

(autoplot(acceptancePCA, data=prsSample, colour="accepted") +
 labs(x=paste("PC1 (", summary(acceptancePCA)$importance[2,1] * 100, "\\%)", sep=""),
      y=paste("PC2 (", summary(acceptancePCA)$importance[2,2] * 100, "\\%)", sep=""), colour="Accepted")
)

tikz(filename="acceptance_pca.tex", width=8, height=3)
(autoplot(acceptancePCA, data=prsSample, colour="accepted") + xlim(-0.0025, 0.0025) + ylim(-0.01, 0.01) +
 labs(x=paste("PC1 (", summary(acceptancePCA)$importance[2,1] * 100, "\\%)", sep=""),
      y=paste("PC2 (", summary(acceptancePCA)$importance[2,2] * 100, "\\%)", sep=""), colour="Accepted")
)
dev.off()
## PCA scatterplot:1 ends here

## Contingency matrices
##    Define function for transforming the data into the contingency matrix:

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:1]]
to_contingency_table <- function(prs_data) {
    ct <- data.frame((prs_data %>% select("accepted" | starts_with("results_")) %>%
                      transmute(accepted, issueTypes=rowSums(.[-1]>0)) %>% group_by(accepted) %>%
                      summarize(across(everything(), tibble::lst(introduced=~sum(.>0), didNotIntroduced=~sum(.==0)))))[,-1])
    colnames(ct) <- c("Issue introduced", "Issue not introduced")
    rownames(ct) <- c("Rejected", "Accepted")
    ct
}
## Contingency matrices:1 ends here



## Define function that will be used to plot results of chi-square test of independence:

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:2]]
chsqt_plot <- function(chsqt) {
    ggplot(data=data.frame(Frequency=c(chsqt$observed[1,1], chsqt$observed[1,2], chsqt$observed[2,1], chsqt$observed[2,2],
                                       chsqt$expected[1,1], chsqt$expected[1,2], chsqt$expected[2,1], chsqt$expected[2,2]),
                           Value=rep(c("Observed", "Expected"), each=4),
                           Quality=rep(c("Issue detected", "Without an issue"), times=4),
                           Acceptance=rep(rep(c("Rejected pull requests", "Accepted pull requests"), each=2), times=2)
                           ), aes(x=Quality, y=Frequency, fill=Value)) + geom_bar(stat="identity", position="dodge") +
        facet_grid(~ Acceptance) + labs(x="Presence of some quality issue", y="Pull request frequency")
}
## Contingency matrices:2 ends here



## Define function for printing chi-square test:

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:3]]
chsqt_print <- function(ct_name, ct) {
    print(ct_name)
    chsqtType <- chisq.test(ct)
    print(chsqtType)
    print("Observed:")
    print(chsqtType$observed)
    print("Expected:")
    print(chsqtType$expected)
    print(cramersV(ct))
    print(paste(rep("-", times=80), collapse=""))
}
## Contingency matrices:3 ends here



## Does an introduction of some code quality issue in the PR affects its acceptance?

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:4]]
qualityCT <- to_contingency_table(prs)

chsqt_print("All PRs", qualityCT)

tikz(filename="acceptance_ct.tex", width=8, height=3)
chsqt_plot(chisq.test(qualityCT))
dev.off()
## Contingency matrices:4 ends here



## Filter PRs that only modified some source code files and test them:

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:5]]
only_modified <- function(data) {
    data %>% filter(modified == linted_and_modified, added == 0, deleted == 0)
}

qualityModCT <- to_contingency_table(only_modified(prs))

chsqt_print("PR's that only modified some source code files", qualityModCT)

tikz(filename="acceptance_mod_ct.tex", width=10, height=6)
chsqt_plot(chisq.test(qualityModCT))
dev.off()
## Contingency matrices:5 ends here



## Test each issue category independently:

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:6]]
for (type in unique(issues$type)) {
    chsqt_print(type, to_contingency_table(prs %>% select(accepted, contains(paste("_", type, "_", sep="")))))
}
## Contingency matrices:6 ends here



## Test each project independently:

## [[file:pr_analysis.org::*Contingency matrices][Contingency matrices:7]]
projects_chsqt <- tibble(Project=character(), pr_count=integer(), p.value=numeric(), V=numeric(),
                         observed.rejected.introduced=numeric(), expected.rejected.introduced=numeric(),
                         Independent=logical(), "Enough observations"=logical())
for (project in unique(prs$project_name)) {
    project_prs <- prs %>% filter(project_name == project)
    ct <- to_contingency_table(project_prs)
    chsqt_print(project, ct)
    chsqt <- chisq.test(ct)
    projects_chsqt <- projects_chsqt %>% add_row(Project=project, pr_count=nrow(project_prs), p.value=chsqt$p.value,
                                                 V=cramersV(ct), observed.rejected.introduced=chsqt$observed[[1]],
                                                 expected.rejected.introduced=chsqt$expected[[1]] - chsqt$observed[[1]],
                                                 Independent=(chsqt$p.value > 0.05),
                                                 "Enough observations"=all(chsqt$expected >= 10))
}

projects_chsqt %>% print(width=Inf)

count(projects_chsqt %>% filter(`Enough observations` == TRUE))

projects_chsqt %>% filter(`Enough observations` == TRUE) %>% print(width=Inf)

count(projects_chsqt %>% filter(`Enough observations` == TRUE, p.value < 0.05))

projects_chsqt %>% filter(`Enough observations` == TRUE, p.value < 0.05) %>% print(width=Inf)

projects_chsqt %>% filter(`Enough observations` == TRUE, p.value < 0.05) %>% summarize_all(mean)
## Contingency matrices:7 ends here

## ROC curves and AUCs
##    Retrieve classifiers metrics:

## [[file:pr_analysis.org::*ROC curves and AUCs][ROC curves and AUCs:1]]
import_classification_metrics <- function(path) {
    files <- list.files(path=path, pattern="*.csv")
    metrics <- lapply(files, (function (file) read.csv(paste(path, file, sep=""), header=TRUE, sep = ",", check.names=FALSE)))
    names(metrics) = lapply(files, function (file) sub("_ruleid.csv", "", file))
    bind_rows(metrics, .id="Classifier")
}

classificationMetrics <- import_classification_metrics("classification/Metrics/")

classificationMetrics

classificationMetrics %>% summarise_all(mean)

classificationMetricsIntroduced <- import_classification_metrics("classification_introduced/Metrics/")

classificationMetricsIntroduced

classificationMetricsFixed <- import_classification_metrics("classification_fixed/Metrics/")

classificationMetricsFixed
## ROC curves and AUCs:1 ends here



## Create table with classification metrics:

## [[file:pr_analysis.org::*ROC curves and AUCs][ROC curves and AUCs:2]]
classTable <- classificationMetrics %>% select("Classifier", "AUC_mean", "Precision_mean", "Recall_mean", "MCC_mean", "F1_mean") %>%
    rename_with(stringr::str_replace, pattern="_mean", replacement="") %>% rename("F-Measure"=F1) %>%
    mutate_if(is.numeric, ~(as.character(round(.x / 100, digits=4))))

print(xtable(classTable, type="latex", align=c("l", "|p{3cm}", "p{1.2cm}", "p{1.2cm}", "p{1.2cm}", "p{1.2cm}", "p{1.5cm}|")),
      file="classification_metrics.tex", include.rownames=FALSE,
      add.to.row=list(pos=list(0), command=c("\\hline")), floating=FALSE)
## ROC curves and AUCs:2 ends here



## Import ROC curves:

## [[file:pr_analysis.org::*ROC curves and AUCs][ROC curves and AUCs:3]]
import_roc_curves <- function(path, metrics) {
    files <- list.files(path=path, pattern="*.csv")
    rocs <- lapply(files, (function (file) read.csv(paste(path, file, sep=""), header=TRUE, sep = ",", check.names=FALSE)))
    names(rocs) = (metrics %>% transmute(names=gsub("%", "\\\\%", stringr::str_c(Classifier, " (AUC=",
                                                                              as.character(percent(AUC_mean/100)), ")")
                                                    )))$names
    rocs
}

classificationROCs <- import_roc_curves("classification/AUCs/values/", classificationMetrics)

classificationIntroducedROCs <- import_roc_curves("classification_introduced/AUCs/values/", classificationMetricsIntroduced)

classificationFixedROCs <- import_roc_curves("classification_fixed/AUCs/values/", classificationMetricsFixed)
## ROC curves and AUCs:3 ends here



## Plot ROC curves:

## [[file:pr_analysis.org::*ROC curves and AUCs][ROC curves and AUCs:4]]
plot_roc_curves <- function(rocs) {
    ggplot(bind_rows(rocs, .id="Classifier"), aes(x=mean_fpr, y=mean_tpr, colour=Classifier)) + geom_line() +
        geom_abline(intercept=0, slope=1, linetype="dashed") + xlab("False Positive Rate") + ylab("True Positive Rate")
}

tikz(filename="roc_curves.tex", width=8, height=3)
plot_roc_curves(classificationROCs)
dev.off()

plot_roc_curves(classificationIntroducedROCs)

plot_roc_curves(classificationFixedROCs)
## ROC curves and AUCs:4 ends here

## Research Question 4
##   Retrieve regression metrics:

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:1]]
import_regression_metrics <- function(path) {
    files <- list.files(path=path, pattern="*_metrics.csv")
    metrics <- lapply(files, (function (file) read.csv(paste(path, file, sep=""), header=TRUE, sep = ",", check.names=FALSE)))
    names(metrics) = lapply(files, function (file) sub("_metrics.csv", "", file))
    bind_rows(metrics, .id="Regressor")
}

regressionMetrics <- import_regression_metrics("regression/")

regressionMetrics

regressionMetricsIntroduced <- import_regression_metrics("regression_introduced/")

regressionMetricsIntroduced

regressionMetricsFixed <- import_regression_metrics("regression_fixed/")

regressionMetricsFixed
## Research Question 4:1 ends here



## Create table with regression metrics:

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:2]]
regTable <- regressionMetrics %>% mutate(across(c(EV, R2), ~(round(.x, digits=4)))) %>%
    mutate(across(c(MAE, MSE), ~(round(.x, digits=0)))) %>% rename("$R^2$"=R2) %>% mutate_all(as.character)

print(xtable(regTable, type="latex", align=c("l", "|p{3cm}", "p{1.5cm}", "p{2.5cm}", "p{1.2cm}", "p{1.2cm}|")),
      file="regression_metrics.tex", include.rownames=FALSE, sanitize.text.function=function(x){x},
      add.to.row=list(pos=list(0), command=c("\\hline")), floating=FALSE)
## Research Question 4:2 ends here



## Import predicted/actual values (regression):

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:3]]
import_regression_plots <- function(path, metrics) {
    files <- list.files(path=path, pattern="*_predicted.csv")
    curves <- lapply(files, (function (file) read.csv(paste(path, file, sep=""), header=TRUE, sep = ",", check.names=FALSE)))
    names(curves) = metrics$Regressor
    curves
}

regressionPlots <- import_regression_plots("regression/", regressionMetrics)

regressionPlotsIntroduced <- import_regression_plots("regression_introduced/", regressionMetricsIntroduced)

regressionPlotsFixed <- import_regression_plots("regression_fixed/", regressionMetricsFixed)
## Research Question 4:3 ends here



## Plot regression curves (predicted vs actual):

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:4]]
plot_regression_plots <- function(curves) {
    ggplot(bind_rows(curves, .id="Regressor"), aes(x=Predicted, y=Actual, colour=Regressor)) + geom_point() +
        geom_abline(intercept=0, slope=1, linetype="dashed")
}

tikz(filename="regression_predicted.tex", width=10, height=4)
plot_regression_plots(regressionPlots)
dev.off()

plot_regression_plots(regressionPlotsIntroduced)

plot_regression_plots(regressionPlotsFixed)
## Research Question 4:4 ends here



## Plot density of absolute error:

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:5]]
tikz(filename="regression_absolute_error.tex", width=8, height=3)
ggplot(bind_rows(regressionPlots, .id="Regressor"), aes(x=(abs(Actual - Predicted) / 2629800), colour=Regressor)) +
    geom_density() + geom_vline(xintercept=0, linetype="dashed") +
    geom_vline(aes(xintercept=(MAE / 2629800), colour=Regressor), data=regressionMetrics, linetype="dashed") +
    xlim(-0.5, 8) + labs(x="Absolute Error (months)", y="Density")
dev.off()
## Research Question 4:5 ends here



## Compute $R^2$ when considering only PRs that was closed within a month:

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:6]]
bind_rows(regressionPlots, .id="Regressor") %>% group_by(Regressor) %>% filter(Actual < 2629800) %>%
    summarize(R2=cor(Predicted, Actual) ^ 2)
## Research Question 4:6 ends here



## Print the percentage of PRs that was closed within a month:

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:7]]
percent(nrow(prs %>% filter(time_opened < 2629800))/nrow(prs))
## Research Question 4:7 ends here



## Print summary about all regression metrics:

## [[file:pr_analysis.org::*Research Question 4][Research Question 4:8]]
regressionMetrics %>% summarize(across(where(is.double), ~mean(.x)))

regressionMetricsIntroduced %>% summarize(across(where(is.double), ~mean(.x)))

regressionMetricsFixed %>% summarize(across(where(is.double), ~mean(.x)))
## Research Question 4:8 ends here

## Research Question 5
##   Set the working directory:

## [[file:pr_analysis.org::*Research Question 5][Research Question 5:1]]
setwd('~/Documents/master/results/')
## Research Question 5:1 ends here



## Import all pull requests:

## [[file:pr_analysis.org::*Research Question 5][Research Question 5:2]]
prsAll <- list("C/C++" = import_prs("c_cpp/data.csv"), "Haskell" = import_prs("haskell/data.csv"),
               "Java" = import_prs("java/data.csv"), "Kotlin" = import_prs("kotlin/data.csv"),
               "Python" = import_prs("python/data.csv"))
## Research Question 5:2 ends here

## Chi-square tests
##    Import pull requests from all languages and run chi-square tests:

## [[file:pr_analysis.org::*Chi-square tests][Chi-square tests:1]]
chisqtAll <- prsAll %>% map(~tribble(~V, ~p, ~Type,
                            #
                            cramersV(to_contingency_table(.x)),
                            chisq.test(to_contingency_table(.x))$p.value,
                            "All",
                            #
                            cramersV(to_contingency_table(only_modified(.x))),
                            chisq.test(to_contingency_table(only_modified(.x)))$p.value,
                            "Filtered")
                            ) %>% bind_rows(.id = "Language") %>% mutate(Independent = (p > 0.05))
## Chi-square tests:1 ends here



## Plot the Cramer's V for all languages:

## [[file:pr_analysis.org::*Chi-square tests][Chi-square tests:2]]
tikz(filename="all_cramers_v.tex", width=8, height=4)
chisqtAll %>% ggplot(aes(x=Type, y=V, fill=Independent)) + geom_bar(stat="identity", position="dodge") +
    facet_grid(~ Language) + ylab("Cramer's V")
dev.off()
## Chi-square tests:2 ends here

## Classification
##    Import classification metrics for all languages:

## [[file:pr_analysis.org::*Classification][Classification:1]]
allClassificationMetrics <- bind_rows(list(
    "C/C++" = import_classification_metrics("c_cpp/classification/Metrics/"),
    "Haskell" = import_classification_metrics("haskell/classification/Metrics/"),
    "Java" = import_classification_metrics("java/classification/Metrics/"),
    "Kotlin" = import_classification_metrics("kotlin/classification/Metrics/"),
    "Python" = import_classification_metrics("python/classification/Metrics/")), .id="Language")
## Classification:1 ends here



## Create box plot with /AUC mean/ for each language:

## [[file:pr_analysis.org::*Classification][Classification:2]]
tikz(filename="all_auc.tex", width=8, height=4)
allClassificationMetrics %>% ggplot(aes(x=Language, y=AUC_mean)) + labs(y="AUC mean") +
    geom_boxplot() + geom_point(aes(col=Classifier), size=5)
dev.off()
## Classification:2 ends here

## Regression
##    Import regression metrics for all languages:

## [[file:pr_analysis.org::*Regression][Regression:1]]
allRegressionMetrics <- bind_rows(list(
    "C/C++" = import_regression_metrics("c_cpp/regression/"),
    "Haskell" = import_regression_metrics("haskell/regression/"),
    "Java" = import_regression_metrics("java/regression/"),
    "Kotlin" = import_regression_metrics("kotlin/regression/"),
    "Python" = import_regression_metrics("python/regression/")), .id="Language")
## Regression:1 ends here



## Create box plot with /MAE/ for each language:

## [[file:pr_analysis.org::*Regression][Regression:2]]
tikz(filename="all_mae.tex", width=8, height=4)
allRegressionMetrics %>% mutate_at("MAE", function(mae) mae / 86400) %>% ggplot(aes(x=Language, y=MAE)) + labs(y="MAE (days)") +
    geom_boxplot() + geom_point(aes(col=Regressor), size=5)
dev.off()
## Regression:2 ends here



## Compute the percentage of PRs that was close within first two weeks:

## [[file:pr_analysis.org::*Regression][Regression:3]]
prsAll %>% map(~(nrow(.x %>% filter(time_opened < 1209600)) / nrow(.x))) %>% print
## Regression:3 ends here

## Summary
##   Print the total number of PRs for all languages:

## [[file:pr_analysis.org::*Summary][Summary:1]]
Reduce("+", prsAll %>% map(~nrow(.x)))
## Summary:1 ends here



## Print the number of accepted PRs:

## [[file:pr_analysis.org::*Summary][Summary:2]]
Reduce("+", prsAll %>% map(~nrow(.x %>% filter(accepted == TRUE))))
## Summary:2 ends here



## Print the number of PRs that introduced some issue:

## [[file:pr_analysis.org::*Summary][Summary:3]]
Reduce("+", prsAll %>% map(~(.x %>% mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")),
                                                                         ~if_else(.x < 0, 0L, .x))),
                                           fixed=rowSums(mutate_all(select(., starts_with("results_")),
                                                                    ~if_else(.x > 0, 0L, -.x)))) %>% filter(introduced > 0) %>%
                             nrow)))
## Summary:3 ends here



## Print the number of PRs that fixed some issue:

## [[file:pr_analysis.org::*Summary][Summary:4]]
Reduce("+", prsAll %>% map(~(.x %>% mutate(introduced=rowSums(mutate_all(select(., starts_with("results_")),
                                                                         ~if_else(.x < 0, 0L, .x))),
                                           fixed=rowSums(mutate_all(select(., starts_with("results_")),
                                                                    ~if_else(.x > 0, 0L, -.x)))) %>% filter(fixed > 0) %>%
                             nrow)))
## Summary:4 ends here

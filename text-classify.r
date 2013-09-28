library("Matrix")
library("RWeka")
library("parallel")

EMAILS_DIR = "messages-dist"
STOPWORDS = "stopwords/english"
LOG = paste("logs/Log -", date())

log.message <- function(message)
{
    print(message)
    #cat(paste(message, "\n", sep = ""), file = LOG, append = TRUE)
}

word.cache.size <- 0
make.words.from.email <- function(email.name, email.file, stopwords)
{
    cache.key <- paste("cached.words.for", email.name, sep = ".")
    if( exists(cache.key) )
    {
        return(get(cache.key))
    }

    email.lines <- readLines(email.file)
    tokens <- Map(tolower, WordTokenizer(email.lines))
    sanitized <- Map(function(t) { gsub("[\\'\"_*-<>=|%{}~^]", "", t) }, tokens)
    stemmed <- LovinsStemmer(sanitized[!sanitized %in% stopwords])
    words <- stemmed[nchar(stemmed) > 2]

    # cache results
    assign("word.cache.size", word.cache.size + 1, envir = .GlobalEnv)
    assign(cache.key, words, envir = .GlobalEnv)

    return(words)
}

compute.full.dictionary <- function(email.files, stopwords)
{
    # all unique words
    all.words <- Reduce(
        function(all.words.so.far, ef)
        {
            email <- basename(ef)
            words.in.email <- make.words.from.email(email, ef, stopwords)
            return(union(all.words.so.far, words.in.email))
        }, email.files, init = c())

    return(all.words)
}

compute.feature.words <- function(raw.dictionary, count.emails.with.word, total.emails)
{
    too.infrequent <- count.emails.with.word[count.emails.with.word <= 0.01 * total.emails]
    log.message(paste("Found", length(too.infrequent), "infrequent words"))

    feature.words <- setdiff(raw.dictionary, names(too.infrequent))
    return(feature.words)
}

compute.feature.vectors <- function()
{
    log.message("Starting up!")

    stopwords <- readLines(STOPWORDS)
    email.files <- Sys.glob(file.path(EMAILS_DIR, "*.txt"))

    raw.dictionary <- compute.full.dictionary(email.files, stopwords)
    log.message(paste("Found", length(raw.dictionary), "unique words"))
    log.message(paste("Cached words for", word.cache.size, "emails"))

    # calculate how many emails contain a word
    count.emails.with.word <- rep(0, length(raw.dictionary))
    names(count.emails.with.word) <- raw.dictionary

    progress <- 0
    for( ef in email.files )
    {
        email <- basename(ef)
        words.in.email <- make.words.from.email(email, ef, stopwords)

        # XXX this already eliminates duplicates for some reason
        count.emails.with.word[words.in.email] <-
            count.emails.with.word[words.in.email] + 1

        progress <- progress + 1
        if( progress %% 100 == 0 )
        {
            log.message(paste("count.emails.with.word for", progress, "emails"))
        }
    }

    # decide which words will end up being features
    feature.words <- compute.feature.words(raw.dictionary, count.emails.with.word, length(email.files))
    log.message(paste("Ended up with", length(feature.words), "words"))
    remove(raw.dictionary)

    # calculate the idf (inverse document frequency) for each feature word
    count.emails.with.word <- count.emails.with.word[feature.words]
    idf <- log(length(email.files) / count.emails.with.word)
    remove(count.emails.with.word)

    # compute tf, the term frequency (number of times each word appears
    # in each email)
    tf <- Matrix(0, nrow = length(email.files), ncol = length(feature.words))
    rownames(tf) <- Map(basename, email.files)
    colnames(tf) <- feature.words

    progress <- 0
    for( ef in email.files )
    {
        email <- basename(ef)
        words.in.email <- make.words.from.email(email, ef, stopwords)

        count.feature.words.table <-
            table(words.in.email[words.in.email %in% feature.words])
        feature.words.in.email <- rownames(count.feature.words.table)

        count.feature.words <- as.vector(count.feature.words.table)

        tf[email,feature.words.in.email] <-
            tf[email,feature.words.in.email] + count.feature.words

        progress <- progress + 1
        if( progress %% 100  == 0 )
        {
            log.message(paste("tf for", progress, "emails"))
        }
    }

    log.message("Done computing idf and tf!")

    fv.not.normalized <- tf * idf
    fv <- fv.not.normalized / sqrt(rowSums(fv.not.normalized ^ 2))

    # export results to global env
    assign("fv", fv, envir = .GlobalEnv)
    assign("feature.words", feature.words, envir = .GlobalEnv)
    assign("tf", tf, envir = .GlobalEnv)
    assign("idf", idf, envir = .GlobalEnv)
}

do.kmeans <- function(k.values)
{
    log.message("Beginning kmeans")
    kmeans.results <- simplify2array(mclapply(k.values,
        function(k)
        {
            log.message(paste("kmeans for", k, "clusters"))
            kmeans(fv, k, nstart = 20, iter.max = 100)
        }, mc.cores = 8))

    # export results to global env
    assign("kmeans.results", kmeans.results, envir = .GlobalEnv)
}

find.centroids.and.closest.emails <- function(k, centers, clusters)
{
    centroids.and.closest <- simplify2array(mclapply(1:k,
        function(k)
        {
            log.message(paste("Finding centroid for cluster", k))
            center.for.cluster <- centers[k,]
            emails.in.cluster <- names(clusters[clusters == k])

            distances <- sapply(
                emails.in.cluster,
                function(email)
                {
                    fv.for.email <- fv[email,]
                    return(sqrt(sum((fv.for.email - center.for.cluster) ^ 2)))
                }, USE.NAMES = TRUE)

            distances <- sort(distances)
            four.min.dist <- distances[(length(distances)-3):length(distances)]
            return(names(distances[distances <= four.min.dist[4]])[1:4])
        }, mc.cores = 8))

    # export results to global env
    assign("email.centroids", centroids.and.closest[1,], envir = .GlobalEnv)
    assign("closest.emails", centroids.and.closest[-1,], envir = .GlobalEnv)
}

main <- function()
{
    compute.feature.vectors()
}

main()

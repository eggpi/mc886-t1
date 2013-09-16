library("RWeka")

EMAILS_DIR = "../../dados/messages-dist"
STOPWORDS = "../../dados/stopwords/english"
LOG = paste("logs/Log -", date())

log.message <- function(message)
{
    cat(paste(message, "\n", sep = ""), file = LOG, append = TRUE)
}

word.cache.size <- 0
make.words.from.email <- function(email.name, email, stopwords)
{
    cache.key <- paste("cached.words.for", email.name, sep = ".")
    if( exists(cache.key) )
    {
        return(get(cache.key))
    }

    tokens <- Map(tolower, WordTokenizer(email))
    sanitized <- Map(function(t) { gsub("[\\'\"_*-<>=|%]", "", t) }, tokens)
    stemmed <- LovinsStemmer(setdiff(sanitized, stopwords))
    words <- stemmed[nchar(stemmed) > 2]

    if (length(email) > 20)
    {
        assign("word.cache.size", word.cache.size + 1, envir = .GlobalEnv)
        assign(cache.key, words, envir = .GlobalEnv)
    }

    return(words)
}

compute.full.dictionary <- function(email.files, stopwords)
{
    # all unique words we're going to use as features
    all.words <- c()

    for( ef in email.files )
    {
        email <- basename(ef)
        words.in.email <- make.words.from.email(email, readLines(ef), stopwords)
        all.words <- union(all.words, words.in.email)
    }

    return(all.words)
}

compute.feature.words <- function(raw.dictionary, count.emails.with.word)
{
    too.infrequent <- count.emails.with.word[count.emails.with.word <= 2]
    log.message(paste("Found", length(too.infrequent), "infrequent words"))

    feature.words <- setdiff(raw.dictionary, names(too.infrequent))
    return(feature.words)
}

compute.idf <- function(count.emails.with.word, feature.words, email.count)
{
    idf <- rep(0, length(feature.words))
    names(idf) <- feature.words
    for( w in feature.words )
    {
        idf[w] <- log(email.count / count.emails.with.word[w])
    }

    return(idf)
}

main <- function()
{
    log.message("Starting up!")

    stopwords <- readLines(STOPWORDS)
    email.files <- Sys.glob(file.path(EMAILS_DIR, "*.txt"))

    raw.dictionary <- compute.full.dictionary(email.files, stopwords)
    log.message(paste("Found", length(raw.dictionary), "unique words"))
    log.message(paste("Cached words for", word.cache.size, "emails"))

    # how many emails contain a word
    count.emails.with.word <- rep(0, length(raw.dictionary))
    names(count.emails.with.word) <- raw.dictionary

    # compute words.in.email and use it to calculate tf
    progress <- 0
    for( ef in email.files )
    {
        email <- basename(ef)
        words.in.email <- make.words.from.email(email, readLines(ef), stopwords)
        unique.words <- unique(words.in.email)

        for( w in unique.words )
        {
            count.emails.with.word[w] <- count.emails.with.word[w] + 1
        }

        progress <- progress + 1
        if( progress %% 100 == 0 )
        {
            log.message(paste("count.emails.with.word for", progress, "emails"))
        }
    }

    # decide which words will end up being features
    feature.words <- compute.feature.words(raw.dictionary, count.emails.with.word)
    remove(raw.dictionary)
    log.message(paste("Ended up with", length(feature.words), "words"))

    # calculate the idf (inverse document frequency) for each feature word
    idf <- compute.idf(count.emails.with.word, feature.words, length(email.files))
    remove(count.emails.with.word)

    # compute word.count.in.email (how many times a word was used in an email)
    # and use it to calculate idf
    word.count.in.email <- matrix(
        rep(0, length(email.files) * length(feature.words)),
        nrow = length(email.files), ncol = length(feature.words))

    rownames(word.count.in.email) <- Map(basename, email.files)
    colnames(word.count.in.email) <- feature.words

    progress <- 0
    for( ef in email.files )
    {
        email <- basename(ef)
        words.in.email <- make.words.from.email(email, readLines(ef), stopwords)
        feature.words.in.email <- intersect(words.in.email, feature.words)

        for( w in feature.words.in.email )
        {
            word.count.in.email[email, w] <- word.count.in.email[email, w] + 1
        }

        progress <- progress + 1
        if( progress %% 100  == 0 )
        {
            log.message(paste("word.count.in.email for", progress, "emails"))
        }
    }

    # calculate the term frequency for each feature word
    tf <- word.count.in.email[,colnames(word.count.in.email) %in% feature.words]
    remove(word.count.in.email)

    log.message("All done!")

    # export our results to the workspace
    assign("feature.words", feature.words, envir = .GlobalEnv)
    assign("idf", idf, envir = .GlobalEnv)
    assign("tf", tf, envir = .GlobalEnv)
}

main()

using Markdown

macro includeshow(filepath)
    :(
        include($filepath);
        content = read($filepath, String);
        Markdown.parse("```julia\n$(content)\n```")
    )
end
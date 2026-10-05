# Keep the stable install path so an already-running shell survives upgrades.
set -g __richengine_completion_source (builtin realpath -s -- (status filename))

function __richengine_models
    # The helper beside the script's current target, found again each time.
    set -l helper (path dirname -- (path resolve -- $__richengine_completion_source))/models
    test -x $helper
    and $helper
end

# As in Bash and Zsh: only the value of --model, and only after `richengine serve`.
function __richengine_model_value
    set -l words (commandline -opc)
    test "$words[2]" = serve
    and not string match -qr -- '^-[^=]*$' (commandline -ct)
end

complete -c richengine -f
complete -c richengine -n 'test (count (commandline -opc)) -eq 1' -a 'serve models status flags doctor disk claude codex opencode hermes pi'
complete -c richengine -n __richengine_model_value -l model -x -a '(__richengine_models)'

#ifndef COMPAT_H
#define COMPAT_H

#ifdef _WIN32
    // On est sur Windows : on utilise les fonctions Microsoft
    #include <stdio.h>
    // Si tu veux vraiment utiliser sscanf_s, rien à faire, mais 
    // pour que le code soit portable, on fait souvent l'inverse :
#else
    // On est sur Linux/CachyOS : on mappe les noms Windows vers les noms POSIX
    #include <stdio.h>
    
    #define _popen popen
    #define _pclose pclose
    
    // Pour sscanf_s, on le redirige vers le sscanf standard
    // Attention : sscanf_s attend parfois des arguments de taille en plus, 
    // sscanf les ignorera ou plantera si tu ne fais pas gaffe.
    #define sscanf_s sscanf
#endif

#endif
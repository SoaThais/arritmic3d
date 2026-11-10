/**
 * ARRITMIC3D
 *
 * (C) CoMMLab-UV 2023
 * */

#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <Eigen/Dense>

#include "basic_tissue.h"
#include "node.h"
#include "error.h"

#include <string>
#include <iomanip>

using std::vector;

/**
 * @brief Save the state of the tissue in a VTK file for visualization.
 * @param filename Name of the file to save. It should end with .vtk.
 */
template <typename APM,typename CVM>
void BasicTissue<APM,CVM>::SaveVTK(const std::string & filename) const
{
    std::ofstream vtk_file;
    vtk_file.open(filename);
    if(!vtk_file)
    {
        LOG::Error(true, "Could not open file " + filename + " for writing.");
        return;
    }
    // Write the header
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Cardiac Tissue\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET RECTILINEAR_GRID\n";
    vtk_file << "DIMENSIONS " << tissue_geometry.size_x << " " << tissue_geometry.size_y << " " << tissue_geometry.size_z << std::endl;
    vtk_file << "X_COORDINATES " << tissue_geometry.size_x << " float" << std::endl;
    for(int i = 0; i < tissue_geometry.size_x; i++)
    {
        vtk_file << tissue_geometry.origin[0] + i*tissue_geometry.dx << " ";
        if((i+1) % 10 == 0)
            vtk_file << "\n";
    }
    vtk_file << std::endl;
    vtk_file << "Y_COORDINATES " << tissue_geometry.size_y << " float" << std::endl;
    for(int i = 0; i < tissue_geometry.size_y; i++)
    {
        vtk_file << tissue_geometry.origin[1] + i*tissue_geometry.dy << " ";
        if((i+1) % 10 == 0)
            vtk_file << "\n";
    }
    vtk_file << std::endl;
    vtk_file << "Z_COORDINATES " << tissue_geometry.size_z << " float" << std::endl;
    for(int i = 0; i < tissue_geometry.size_z; i++)
    {
        vtk_file << tissue_geometry.origin[2] + i*tissue_geometry.dz << " ";
        if((i+1) % 10 == 0)
            vtk_file << "\n";
    }
    vtk_file << std::endl;

    // Write the data
    vtk_file << "\nPOINT_DATA " << tissue_geometry.size_x * tissue_geometry.size_y * tissue_geometry.size_z << std::endl;
    vtk_file << "SCALARS Type int 1\n";
    vtk_file << "LOOKUP_TABLE default" << std::endl;
    for(int i = 0; i < int(this->size()); i++)
    {
        auto index = tissue_geometry.GetMemIndex_from_GridIndex(i);
        if(index != NO_INDEX)
        {
            vtk_file << int(tissue_nodes[index].type) << " ";
        }
        else
        {
            vtk_file << int(CELL_TYPE_VOID) << " ";
        }
        if((i+1) % 10 == 0)
            vtk_file << "\n";
    }
    vtk_file << std::endl;

    vtk_file << "SCALARS State int 1\n";
    vtk_file << "LOOKUP_TABLE default" << std::endl;
    for(int i = 0; i < int(this->size()); i++)
    {
        auto index = tissue_geometry.GetMemIndex_from_GridIndex(i);
        if(index != NO_INDEX)
        {
            vtk_file << int(tissue_nodes[index].GetState(tissue_time) ) << " ";
        }
        else
        {
            vtk_file << 0 << " ";
        }
        if((i+1) % 10 == 0)
            vtk_file << "\n";
    }
    vtk_file << std::endl;

    vtk_file.close();
}

/**
 * @brief Save the state of the tissue in a VTK file for visualization.
 * @param filename Name of the file to save. It should end with .vtk.
 * @param data_id Bitwise OR of NodeDataId values to select which data to save.
 * @param binary If true, the data will be saved in binary format.
 */
template <typename APM,typename CVM>
void BasicTissue<APM,CVM>::SaveVTKPoints(const std::string & filename, const int data_id, bool binary) const
{
    std::ofstream vtk_file;
    if(binary)
        vtk_file.open(filename, std::ios::binary);
    else
        vtk_file.open(filename);
    if(!vtk_file)
    {
        LOG::Error(true, "Could not open file " + filename + " for writing.");
        return;
    }
    // Write the header
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Cardiac Tissue\n";
    if(binary)
        vtk_file << "BINARY\n";
    else
        vtk_file << "ASCII\n";
    vtk_file << "DATASET POLYDATA\n";

    // Write the points
    vtk_file << "POINTS " << this->tissue_nodes.size() << " float\n";
    for(size_t i = 0; i < this->tissue_nodes.size(); i++)
    {
        auto grid_index = tissue_nodes[i].id;
        auto coords = tissue_geometry.GetCoords_from_GridIndex(grid_index);
        auto real_coords = tissue_geometry.GetRealCoords_from_Coords(coords);
        if(binary)
        {
            vtk_file.write(reinterpret_cast<const char*>(real_coords.data()), sizeof(float) * 3);
        }
        else
        {
            vtk_file << real_coords[0] << " " << real_coords[1] << " " << real_coords[2] << "\n";
        }
    }

    if(!binary)
        vtk_file << std::endl;

    // Write the data
    vtk_file << "POINT_DATA " << this->tissue_nodes.size() << std::endl;

    if(data_id & NodeDataId::ID)
    {
        vtk_file << "SCALARS ID unsigned_int 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
                vtk_file.write(reinterpret_cast<const char*>(&tissue_nodes[i].id), sizeof(unsigned int));
            else
                vtk_file << int(tissue_nodes[i].id) << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

    if(data_id & NodeDataId::TYPE)
    {
        vtk_file << "SCALARS Type unsigned_short 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
                vtk_file.write(reinterpret_cast<const char*>(&tissue_nodes[i].type), sizeof(unsigned short));
            else
                vtk_file << int(tissue_nodes[i].type) << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

    if(data_id & NodeDataId::STATE)
    {
        vtk_file << "SCALARS State char 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
            {
                char state = char(tissue_nodes[i].GetState(tissue_time));
                vtk_file.write(reinterpret_cast<const char*>(&state), sizeof(char));
            }
            else
                vtk_file << int(tissue_nodes[i].GetState(tissue_time) ) << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

    if(data_id & NodeDataId::BEAT)
    {
        vtk_file << "SCALARS Beat int 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
            {
                int beat = int(tissue_nodes[i].GetBeat());
                vtk_file.write(reinterpret_cast<const char*>(&beat), sizeof(int));
            }
            else
                vtk_file << int(tissue_nodes[i].GetBeat() ) << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

     if(data_id & NodeDataId::APD)
    {
        vtk_file << "SCALARS APD float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
            {
                float apd = tissue_nodes[i].apd_model.getAPD();
                vtk_file.write(reinterpret_cast<const char*>(&apd), sizeof(float));
            }
            else
                vtk_file << tissue_nodes[i].apd_model.getAPD() << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

     if(data_id & NodeDataId::CV)
    {
        vtk_file << "SCALARS CV float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
                vtk_file.write(reinterpret_cast<const char*>(&tissue_nodes[i].conduction_vel), sizeof(float));
            else
                vtk_file << tissue_nodes[i].conduction_vel << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

     if(data_id & NodeDataId::LAT)
    {
        vtk_file << "SCALARS LAT float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
            {
                float lat = tissue_nodes[i].apd_model.getActivationTime();
                vtk_file.write(reinterpret_cast<const char*>(&lat), sizeof(float));
            }
            else
                vtk_file << tissue_nodes[i].apd_model.getActivationTime() << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }

     if(data_id & NodeDataId::LIFE)
    {
        vtk_file << "SCALARS Life float 1\n";
        vtk_file << "LOOKUP_TABLE default\n";
        for(int i = 0; i < int(this->tissue_nodes.size()); i++)
        {
            if(binary)
            {
                float life = tissue_nodes[i].apd_model.getLife(tissue_time);
                vtk_file.write(reinterpret_cast<const char*>(&life), sizeof(float));
            }
            else
                vtk_file << tissue_nodes[i].apd_model.getLife(tissue_time) << "\n";
        }
        if(!binary)
            vtk_file << std::endl;
    }


    vtk_file.close();
}

/**
 * @brief Save the activation times history in an ACM file.
 * @param output_dir Directory to save the file.
 * @param prefix Prefix of the filename.
 */
template <typename APM, typename CVM>
void BasicTissue<APM, CVM>::SaveActivationTimesACM (const std::string& output_dir, const std::string& prefix) const {
    
    std::string filepath = output_dir + "/" + prefix + ".acm";
    std::ofstream out(filepath);

    if (!out.is_open()) {
        LOG::Error(true, "[SaveACM] Erro ao abrir arquivo para escrita: " + filepath);
        return;
    }

    // Iterar sobre todos os nós da malha
    for (size_t i = 0; i < this->tissue_nodes.size(); i++) {
        const auto& node = this->tissue_nodes[i];
        
        // Pular nós inativos/buracos
        if (node.type == CELL_TYPE_VOID) continue;

        // Extrair coordenadas reais usando a geometria (idêntico ao SaveVTKPoints)
        auto grid_index = node.id;
        auto coords = tissue_geometry.GetCoords_from_GridIndex(grid_index);
        auto real_coords = tissue_geometry.GetRealCoords_from_Coords(coords);

        float cx = real_coords[0];
        float cy = real_coords[1];
        float cz = real_coords[2];

        // Extrair espaçamento (dx, dy, dz) da geometria global
        float dx = tissue_geometry.dx / 2.0f;
        float dy = tissue_geometry.dy / 2.0f;
        float dz = tissue_geometry.dz / 2.0f;

        // Vetores com o histórico de batimentos (têm de ser declarados em node.h)
        const std::vector<float>& ats = node.at_history;
        const std::vector<float>& apds = node.apd_history;
        
        int num_stimulus = ats.size();

        out << std::fixed << std::setprecision(2)
            << cx << "," << cy << "," << cz << "," 
            << dx << "," << dy << "," << dz << " " 
            << num_stimulus << " [ ";

        for (int s = 0; s < num_stimulus; ++s) {
            out << std::setprecision(6) << ats[s] << " ";
        }

        out << "] [ ";

        for (int s = 0; s < num_stimulus; ++s) {
            // Proteção para não aceder fora dos limites do array
            if (s < apds.size()) {
                out << apds[s] << " ";
            }
        }

        out << "]\n";
    }

    out.close();
}

#endif
